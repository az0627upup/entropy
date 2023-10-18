d = {"红色": 12, "绿色": 32, "红色": 19, "黄色": 23, "绿色":15, "红色":89, "绿色":56, "黄色":55, "黄色":5}
# 给定的字典，使用嵌套字典表示
# 给定的数据，使用列表和元组表示
data = [
    ("红色", 12),
    ("绿色", 32),
    ("红色", 19),
    ("红色", 18),
    ("黄色", 2),
    ("绿色", 15),
    ("绿色", 18),
    ("红色", 89),
    ("绿色", 56),
    ("黄色", 55),
    ("黄色", 2),
    ("黄色", 2),
]

# 使用列表来存储各个类别的元素和对应的下标
color_elements = {}

# 遍历数据，将元素和下标存储在列表中
for index, (color, value) in enumerate(data):
    if color not in color_elements:
        color_elements[color] = []
    color_elements[color].append((value, index))
print(color_elements)

# 找出各类别排名前二的元素的下标
top_two_indices = {}

for color, elements in color_elements.items():
    print(elements)
    elements.sort(reverse=True)
    print(elements)
    top_two_indices[color] = [element[1] for element in elements[-2:]]
print(top_two_indices)
# 输出各类别排名前二的元素的下标
for color, indices in top_two_indices.items():
    print(f"颜色: {color}, 排名前二的元素的下标: {indices}")
