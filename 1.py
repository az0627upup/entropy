# 给定的字典
d = {"红色": 12, "绿色": 32, "红色": 19, "黄色": 23, "绿色":15, "红色":89, "绿色":56, "黄色":55, "黄色":5}
print(d)
# 创建一个字典来存储每种颜色的值和下标
color_values = {}

# 遍历字典，将每种颜色的值和下标存储起来
for index, (color, value) in enumerate(d.items()):
    if color in color_values:
        color_values[color].append((index, value))
    else:
        color_values[color] = [(index, value)]

# 创建一个字典来存储每种颜色排名前两的具体元素
top_two_elements = {}

# 遍历颜色值字典，找出排名前两的元素
for color, values in color_values.items():
    values.sort(key=lambda x: x[1], reverse=True)
    top_two_elements[color] = values[:2]

# 输出排名前两的具体元素
for color, elements in top_two_elements.items():
    print(f"颜色: {color}, 排名前两的具体元素: {elements}")
