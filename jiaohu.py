# coding: utf-8
from math import sin, cos, pi, e, tan, gcd, log, sqrt, factorial
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# img = Image.open(r"D:/python2023/pythonProject/tupian/1.png")
# img.show()

def get_value():
    try:
        print('输入图像路径:', Entry_word.get())
        value = str(eval(Entry_word.get()))
        print(value)
        return value
    except BaseException:
        print('输入出错！')
        return ''
def main():
    global Entry_word, com
    root = tk.Tk()
    root.title("腭中缝成熟度预测软件 V1.0")
    root.geometry('700x300+300+300')
    root.resizable(False, False) # 设置界面是否可以随意拉伸
    arithmetic_polynomials = tk.Label(root, text="输入图像路径:")
    arithmetic_polynomials.grid(row=1, column=0)

    operational = tk.Label(root, text="        输入图像：")
    operational.grid(row=3, column=3)

    img = Image.open("D:/python2023/pythonProject/tupian/1.png")
    photo = ImageTk.PhotoImage(img.resize((100, 100)))
    imgLabel = tk.Label(root, image=photo)
    imgLabel.grid(row=3, column=4)

    # photo = tk.PhotoImage(file="D:/python2023/pythonProject/tupian/1.png")  # file：t图片路径
    # imgLabel = tk.Label(root, image=photo)  # 把图片整合到标签类中
    # #imgLabel.pack(side=tk.RIGHT)  # 自动对齐
    # imgLabel.grid(row=3, column=3)


    Entry_word = tk.Entry(root, width=30)
    Entry_word.grid(row=1, column=1)
    def btn2_cmd(btn=None):
        print("图像路径:%s " % (Entry_word.get()))
        answer_word.delete(0, tk.END)
        answer_word.insert(tk.END, get_value())
    btn2 = tk.Button(
                root,
                text="   输   入   ",
                command=btn2_cmd)
    btn2.grid(row=1, column=2)

    answer = tk.Label(root, text="输出结果:")
    answer.grid(row=4, column=0)

    answer_word = tk.Entry(root, width=10)
    answer_word.grid(row=4, column=1)

    def btn_cmd(btn=None):
        s = """
        A期：腭中缝呈一条较为平直的高密度白线，没有或很少有弯曲。
        
        B期：腭中缝呈一条锯齿状的高密度白线，局部可能有两条相互平行的高密度线条或低密度团块。
        
        C期：腭中缝出现两条相互平行的锯齿状高密度白线，在某些区域被小的低密度团块分隔。
        
        D期：腭中缝从后向前开始钙化，位于后部腭骨的腭中缝与周围骨密度相近，已不可见，上颌骨部分的腭中缝尚未融合，仍为两条锯齿状高密度线。
        
        E期：沿上颌骨的腭中缝已不可见。
        
        """
        tk.messagebox.showinfo('腭中缝A、B、C、D、E五个成熟度时期形态', s)
    btn = tk.Button(
                root,
                text="输出所属成熟度时期介绍说明",
                relief=tk.GROOVE,
                command=btn_cmd)

    btn.grid(row=4, column=2)
    root.mainloop()
if __name__ == "__main__":
    main()D:\python2023\pythonProject\jiaohu.py