"""
1、编写程序，生成一个长度为20元素值均为整数的列表，并将列表中所有的偶数放在一个新列表中。
2、编写程序，使其判断任意给定的列表中的元素是对称的，是对称数组打 True，不是打印 False。
例如  [1，2，0，2，1]，[1，e，2，p，2，e，1]，[4，1，2，3，3，2，1，4]，
这样的数组都是对称数组。
3、编写程序，找出列表
a = [“hello”,“world”, “fly”, “yoyo”, “congratulations”,“color”] 中单词最长的一个。
"""
#1
list_0 = list(range(20))    #20个元素列表
list_1 = list_0             #复制列表0到列表1
print(list_0)               #输出列表0
for i in list_1:            #删除奇数
    list_1.remove(i+1)
list_2 = list_1             #复制列表1
print(list_2)               #输出新列表2

#2
list_3 = list(input("对称数组判断"))          #创建列表
元素 = len(list_3)                            #统计元素个数
for j in range(元素 // 2):                    #一半列表循环
    if list_3[j] != list_3[元素 - 1 - j]:     #元素对应
        print("False")
        break
else:
    print("True")

#3
a = ["hello", "world", "fly", "yoyo", "congratulations", "color"]
最长 = a[0]                       #设最长的单词
for k in range(len(a)-1):         #循环数组元素个数次
    if len(a[k]) < len(a[k+1]):   #比较长度
        最长 = a[k+1]             #留下最长
print(最长)
