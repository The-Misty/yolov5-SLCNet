import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
# 加载YOLOv5模型
weights = 'model/PPLCNetyolo-coco-05-0923-best.pt'  # 你的模型路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = attempt_load(weights)  # 加载模型
model.to(device)  # 将模型放到适当的设备
# 摄像头获取视频流
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # 获取每一帧

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 图像通道转换，并且添加批次维度
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 进行推理
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    for i, det in enumerate(pred):  # for each detection
        if len(det):
            for *xyxy, conf, cls in det:
                # 绘制框
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # 在矩形左上角-10的位置显示置信度，由于rgb_frame是RGB格式的，故这里设置颜色为蓝色
                cv2.putText(rgb_frame, '%.2f' % conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow('frame', rgb_frame)

    # 如果按下'q'键，退出循环
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()