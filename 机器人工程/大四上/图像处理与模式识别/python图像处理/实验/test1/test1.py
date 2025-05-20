import numpy as np
import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())
# 加载图像
image = cv2.imread("sudoku.png")
# 检查图像是否成功加载
if image is None:
 raise ValueError(f"Error loading image: {args['image']}")
# 显示原始图像
cv2.imshow("Original Image", image)
cv2.waitKey(0)
# 转换为灰度图并模糊处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)
# 自适应阈值处理
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
 cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)
# 查找轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# 初始化数独轮廓
puzzleCnt = None
# 遍历所有轮廓，寻找四边形轮廓
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        puzzleCnt = approx
        break
# 如果没有找到数独外框，则抛出错误
if puzzleCnt is None:
    raise Exception("Could not find Sudoku puzzle outline.")
# 在原图上绘制轮廓
output = image.copy()
cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
cv2.imshow("Puzzle Outline", output)
cv2.waitKey(0)
# 对图像进行透视变换
puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
# 显示透视变换后的图像
cv2.imshow("Puzzle Transform", puzzle)
cv2.waitKey(0)
cv2.destroyAllWindows()