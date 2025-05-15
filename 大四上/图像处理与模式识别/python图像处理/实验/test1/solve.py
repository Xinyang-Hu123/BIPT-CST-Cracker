import cv2
import numpy as np
import pytesseract
import imutils
from imutils.perspective import four_point_transform

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Application/Tesseract-OCR/tesseract.exe'  # Update this path if needed

# 预处理图像
def preprocess_image(image_path):
    """
    对数独图像进行预处理，返回二值化图像。
    """
    # 读取图像
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 自适应阈值处理，将其转换为二值图像
    binary = cv2.adaptiveThreshold(~raw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)

    return binary

# 检测网格交点
def detect_grid(binary):
    rows, cols = binary.shape
    scale = 30  # 可以增加缩放比例来加强边缘区域的线条检测

    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded_hor = cv2.erode(binary, kernel_hor, iterations=2)
    dilated_hor = cv2.dilate(eroded_hor, kernel_hor, iterations=3)

    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded_ver = cv2.erode(binary, kernel_ver, iterations=2)
    dilated_ver = cv2.dilate(eroded_ver, kernel_ver, iterations=3)

    intersections = cv2.bitwise_and(dilated_hor, dilated_ver)
    kernel_point = np.ones((3, 3), dtype=np.uint8)
    eroded_points = cv2.erode(intersections, kernel_point, iterations=1)
    return eroded_points


# 获取网格交点坐标
def get_grid_points(eroded_points):
    ys, xs = np.where(eroded_points > 0)
    x_list, y_list = [], []

    # 合并相近的坐标
    xs = np.sort(xs)
    ys = np.sort(ys)

    # 找到具有更大间距的坐标
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > 10:  # 保证有足够的间距来区分
            x_list.append(xs[i])

    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > 10:
            y_list.append(ys[i])

    return x_list, y_list


# 读取数独单元格数字并返回数独的初始状态
def read_sudoku_cells(binary, x_list, y_list, original_image):
    board = [[0 for _ in range(9)] for _ in range(9)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 0, 255)
    thickness = 2

    for i in range(len(y_list) - 1):
        for j in range(len(x_list) - 1):
            cell = binary[y_list[i]+1:y_list[i+1], x_list[j]+1:x_list[j+1]]
            cell = cell[20:-20, 20:-20]  # 裁剪掉边框
            # 如果需要，可以增加适当的修正来保证图像不会被过度裁剪
            cell = cv2.copyMakeBorder(cell, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # OCR读取数字
            number = pytesseract.image_to_string(cell, config='--psm 6 -c tessedit_char_whitelist=123456789')
            number = number.strip()
            if number.isdigit():
                board[i][j] = int(number)
                center_x = (x_list[j] + x_list[j+1]) // 2
                center_y = (y_list[i] + y_list[i+1]) // 2
                cv2.putText(original_image, number, (center_x - 10, center_y + 10), font, font_scale, color, thickness)

    return board


# 数独回溯算法
def is_valid(sudoku, i, j, num):
    """
    检查在给定位置放置数字是否有效。
    """
    # 检查同一行和同一列
    for x in range(9):
        if sudoku[x][j] == num or sudoku[i][x] == num:
            return False

    # 检查3x3子格
    for x in range(3):
        for y in range(3):
            if sudoku[i // 3 * 3 + x][j // 3 * 3 + y] == num:
                return False

    return True

def backtrack(sudoku, i, j):
    """
    使用回溯算法解决数独。
    """
    if sudoku[i][j] != 0:
        if i == 8 and j == 8:
            return True
        elif j == 8:
            return backtrack(sudoku, i + 1, 0)
        else:
            return backtrack(sudoku, i, j + 1)

    for num in range(1, 10):
        if is_valid(sudoku, i, j, num):
            sudoku[i][j] = num
            if i == 8 and j == 8:
                return True
            elif j == 8:
                if backtrack(sudoku, i + 1, 0):
                    return True
            else:
                if backtrack(sudoku, i, j + 1):
                    return True

    sudoku[i][j] = 0
    return False

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

# 主程序
def main(image_path):
    """
    主程序，执行数独识别与解算。
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)

    # 预处理图像
    warped, puzzleCnt = preprocess_image(image_path)
    # 检测网格交点
    eroded_points = detect_grid(warped)

    # 获取交点坐标
    x_list, y_list = get_grid_points(eroded_points)

    # 读取数独单元格并标记到图像上
    board = read_sudoku_cells(warped, x_list, y_list, original_image)

    # 解数独
    if backtrack(board, 0, 0):
        print("Sudoku Solved!")
    else:
        print("No solution exists.")

    # 打印最终的数独解
    for row in board:
        print(row)

    # 标记后的图像
    cv2.imwrite("sudoku_with_numbers.png", original_image)
    print("Marked Sudoku saved as sudoku_with_numbers.png")

# 使用实例
if __name__ == "__main__":
    image_path = "test1/sudoku1.png"  # 替换为您自己的图片路径
    main(image_path)
