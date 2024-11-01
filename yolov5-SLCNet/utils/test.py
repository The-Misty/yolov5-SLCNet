def firstUniqChar(s):
    char_count = {}  # 创建一个字典来存储字符计数
    n = len(s)

    for i in range(n):
        char = s[i]
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    print(len(char_count))
    for i in range(len(char_count)+1):
        if char_count[s[i]] == 1:
            return i

    return -1

# 示例1
input_str1 = "google"
result1 = firstUniqChar(input_str1)
print(result1)  # 输出: 4

# 示例2
input_str2 = "aa"
result2 = firstUniqChar(input_str2)
print(result2)  # 输出: -1
