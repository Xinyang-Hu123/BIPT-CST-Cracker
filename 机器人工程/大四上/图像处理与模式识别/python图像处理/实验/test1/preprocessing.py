import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform

def preprocess_image(image_path):
    """
    预处理输入的数独图像，包括对比度增强和去噪处理。

    参数:
    image_path (str): 数独图像的文件路径

    返回:
    tuple: (puzzle, warped, puzzleCnt)
        - puzzle: 彩色数独图像的俯视图
        - warped: 灰度化数独图像的俯视图
        - puzzleCnt: 数独轮廓的坐标点
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"找不到图片文件: {image_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 对比度增强（直方图均衡化）
    gray = cv2.equalizeHist(gray)

    # 2. 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # 3. 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 4. 轮廓检测
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None
    # 遍历轮廓以找到数独网格
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 如果找到四边形轮廓，认为是数独网格
        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception("无法找到数独网格的轮廓，请确保图像清晰且包含数独。")

    # 5. 透视变换
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))  # 彩色图像
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))  # 灰度图像

    # 6. 再次进行阈值化处理
    # 提升图像质量，使后续数字识别更准确
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    warped = cv2.bitwise_not(warped)

    # 7. 可选：将数独图像缩放为固定大小（例如: 450x450）
    # 如果你希望缩放数独图像以适应某个大小，可以添加如下代码：
    #puzzle_resized = cv2.resize(puzzle, (450, 450))

    return  warped, puzzleCnt

# 测试
if __name__ == "__main__":
    image_path = "./test1/sudoku2.png"  # 替换为你的数独图像路径
    try:
        warped, puzzleCnt = preprocess_image(image_path)
        #cv2.imshow("Resized Sudoku Puzzle", puzzle_resized)  # 显示缩放后的数独俯视图
        cv2.imshow("Warped Binary", warped)  # 显示二值化后的俯视图
        cv2.waitKey(0)
    except Exception as e:
        print(f"发生错误: {e}")
