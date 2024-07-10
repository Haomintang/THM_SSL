
def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

print(cut([i for i in list], 5))
list1 = cut([i for i in list], 5)

# list_names = [f'list_{i}' for i in range(0, 5)]
# print(list_names)

lists_dict = {}
for i in range(5):
    lists_dict[i] = []  # 使用空列表初始化每个键的值
print(lists_dict)

c = 0
for i in list1:
    print(i)
    for a in range(5):
        lists_dict[a].append(i[a])


print(lists_dict[0])


    # 打印字典中的列表，以验证它们已被正确创建和填充

for name, lst in lists_dict.items():
    print(f"{name}: {lst}")
