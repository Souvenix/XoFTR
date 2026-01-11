import cv2
import numpy as np
from pytlsd import lsd
import matplotlib.pyplot as plt


def detect_and_visualize_lines_lsd(image_path, length_threshold=10, show_image=True):
    """
    使用LSD算法检测图像中的线段，并用随机颜色线进行可视化

    Args:
        image_path: 图像路径
        length_threshold: 线段长度阈值
        show_image: 是否显示图像

    Returns:
        image_with_lines: 带有检测线段的图像
        lines: 检测到的线段坐标 [x1, y1, x2, y2]
    """
    # 读取图像 - 以彩色模式读取以保持原图颜色
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为灰度图用于LSD检测
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 使用LSD算法检测线段
    lines = lsd(img_gray)

    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[:4])

            # 计算线段长度
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 根据长度阈值过滤线段
            if line_length >= length_threshold:
                # 用随机颜色绘制线段 (BGR格式)
                color = tuple(np.random.randint(0, 256, size=3).tolist())
                cv2.line(img_bgr, (x1, y1), (x2, y2), color, 18)
                detected_lines.append([x1, y1, x2, y2])

    if show_image:
        plt.figure(figsize=(10, 8))
        # 将BGR转换为RGB以正确显示在matplotlib中
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return img_bgr, detected_lines

# 使用示例
if __name__ == "__main__":
    # 示例1: 从文件路径检测
    image_with_lines, lines = detect_and_visualize_lines_lsd("C:\\Users\\lhk\\Desktop\\1-IM_00006.jpg", length_threshold=100)

    # 示例2: 从numpy数组检测
    # 假设你有一个图像数组
    # image_array = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
    # image_with_lines, lines = detect_lines_from_array(image_array, length_threshold=20)

    print("LSD线段检测函数已定义")
    print("- detect_and_visualize_lines_lsd: 从文件路径检测线段")
    print("- detect_lines_from_array: 从numpy数组检测线段")
