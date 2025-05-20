#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import cv2
import imutils
from skimage.transform import ProjectiveTransform
"""
单应性变换 —— 把图片嵌到广告牌上
作者：webcohort
URL：https://www.zhihu.com/people/webcohort
"""
# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())
# 加载原始图像(广告牌图像)
image = cv2.imread('test2/shutterstock.png')
# 检查图像是否成功加载
if image is None:
 raise ValueError(f"Error loading image: {args['image']}")
# 显示原始图像
cv2.imshow("Original Image", image)
8
cv2.waitKey(0)
# 转换为灰度图并模糊处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)
# 显示模糊处理后的图像
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
# 阈值分割（大津法）
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 显示阈值分割后的图像
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey(0)
# 查找轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# 初始化我们要寻找的轮廓
wantedCnt = None
# 遍历所有轮廓，寻找四边形轮廓
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        wantedCnt = approx
        break
# 如果没有找到想要的外框，则抛出错误
if wantedCnt is None:
 raise Exception("Could not find the outline we wanted!")
# 在原图上绘制轮廓
output = image.copy()
9
cv2.drawContours(output, [wantedCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", output)
cv2.waitKey(0)
# 对图像进行透视变换
im_src = cv2.imread("test2/man_moon.png") # 源图像（要被嵌入到广告牌上的图像）
print(im_src.shape, im_src.dtype, type(im_src))
height, width, dim = im_src.shape
im_dst = image # 目标图像（广告牌图像）
pt = ProjectiveTransform()
src = np.array([[ 0., 0.],
 [height-1, 0.],
 [height-1, width-1],
 [ 0., width-1]])
dst = np.array([[ wantedCnt[0][0][1], wantedCnt[0][0][0]],
 [ wantedCnt[1][0][1], wantedCnt[1][0][0]],
 [ wantedCnt[2][0][1], wantedCnt[2][0][0]],
 [ wantedCnt[3][0][1], wantedCnt[3][0][0]]])
pt.estimate(src, dst)
# 把 im_dst 的每一个点的坐标，变换成[x,y]形式，每一行一个点，保存到 dst_indices 中
x, y = np.mgrid[:im_dst.shape[0], :im_dst.shape[1]] # x 向下，y 向右
dst_indices = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
# 利用 inverse mapping，找到源图像中对应的点
src_indices = np.round(pt.inverse(dst_indices), 0).astype(int)
# 点坐标不能超出源图像的边界
valid_idx = np.where((src_indices[:,0] < height) & (src_indices[:,1] < width) & 
(src_indices[:,0] >= 0) & (src_indices[:,1] >= 0))
dst_indicies_valid = dst_indices[valid_idx]
src_indicies_valid = src_indices[valid_idx]
# 把源图像中的每一个点，复制到目标图像中对应的点
im_dst[dst_indicies_valid[:,0],dst_indicies_valid[:,1]] = im_src[src_indicies_valid[:,0],src_indicies_valid[:,1]]
# 显示最终合成的图像
cv2.imshow("Embedded Image", im_dst)
cv2.waitKey(0)