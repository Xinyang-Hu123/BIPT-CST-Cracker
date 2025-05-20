import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

# 加载图像
image_path = "sudoku.png"
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"找不到叫作 {image_path} 的图片文件.")

# 显示原始图像
cv2.imshow("Original Image", image)
cv2.imwrite("original_image.png", image)  # 保存原图像
cv2.waitKey(0)

# 图像预处理：灰度化和高斯模糊
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)

# 自适应阈值化
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# 反转二值图像
thresh = cv2.bitwise_not(thresh)

# 显示阈值化后的图像
cv2.imshow("Threshold Image", thresh)
cv2.imwrite("threshold_image.png", thresh)  # 保存阈值化后的图像
cv2.waitKey(0)

# 轮廓检测
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 排序轮廓，按面积从大到小
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

puzzleCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 检查是否是四边形轮廓
    if len(approx) == 4:
        puzzleCnt = approx
        break

# 如果没有找到四边形轮廓，则抛出错误
if puzzleCnt is None:
    raise Exception("找不到 {image_path} 的轮廓.")

# 在原图上绘制轮廓以验证结果
output = image.copy()
cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
cv2.imshow("Puzzle Outline", output)
cv2.imwrite("puzzle_outline.png", output)  # 保存轮廓图
cv2.waitKey(0)

# 透视变换
puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

# 显示透视变换后的图像
cv2.imshow("Puzzle Transform", puzzle)
cv2.imwrite("puzzle_transform.png", puzzle)  # 保存透视变换后的图像
cv2.waitKey(0)

cv2.destroyAllWindows()