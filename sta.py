import os

images = os.listdir('D:/python2023/pythonProject/tupian')
imageslist = []

for i in images:
    filepath = 'D:/python2023/pythonProject/tupian/{} 1'.format(i)
    print(filepath)
    f = open('D:/python2023/pythonProject/txt/name.txt', mode='w')  # 打开文件，若文件不存在系统自动创建。
    # 参数name 文件名，mode 模式。
    f.write(filepath)  # write 写入
a = 'ads'
print(a)






# f.writelines(['hello\n','world\n','你好\n','世界\n'])#writelines()函数 会将列表中的字符串写入文件中，但不会自动换行，如果需要换行，手动添加换行符
#                                                     #参数 必须是一个只存放字符串的列表
f.close()              #关闭文件
