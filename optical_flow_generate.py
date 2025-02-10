import os
import cv2
import numpy as np
# import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt

def calculate_optical_flow(image_path1, image_path2, output_path):
    # 读取两个灰度图像
    #img1 = tiff.imread(image_path1).astype(np.float32)
    img1 = np.array(Image.open(image_path1).convert("L"), dtype=np.float32)
    img2 = np.array(Image.open(image_path2).convert("L"), dtype=np.float32)
    #img2 = tiff.imread(image_path2).astype(np.float32)

    # 使用Farneback方法计算光流
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 将光流结果转换为可视化图像
    # 计算光流的幅度和方向
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 将光流幅度归一化到0-255
    flow_magnitude = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 将光流方向转换为彩色图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2  # 角度
    hsv[..., 1] = 255  # 饱和度
    hsv[..., 2] = flow_magnitude  # 明度
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 保存为PNG格式
    cv2.imwrite(output_path, bgr)


def process_cell_folders(root_dir, output_root):
    # 遍历根目录下的每个细胞文件夹
    for cell_folder in os.listdir(root_dir):
        cell_path = os.path.join(root_dir, cell_folder)
        if not os.path.isdir(cell_path):
            continue

        # 创建对应的输出文件夹
        output_cell_path = os.path.join(output_root, cell_folder)
        os.makedirs(output_cell_path, exist_ok=True)

        print(f"Processing cell folder: {cell_folder}")

        # 遍历alive和dead文件夹
        for status_folder in os.listdir(cell_path):
            status_path = os.path.join(cell_path, status_folder)
            if not os.path.isdir(status_path):
                continue

            # 创建对应的输出文件夹
            output_status_path = os.path.join(output_cell_path, status_folder)
            os.makedirs(output_status_path, exist_ok=True)

            print(f"  Processing status folder: {status_folder}")

            # 遍历日期文件夹
            for date_folder in os.listdir(status_path):
                date_path = os.path.join(status_path, date_folder)
                if not os.path.isdir(date_path):
                    continue

                # 创建对应的输出文件夹
                output_date_path = os.path.join(output_status_path, date_folder)
                os.makedirs(output_date_path, exist_ok=True)

                print(f"    Processing date folder: {date_folder}")

                # 遍历每个细胞文件夹
                for cell_subfolder in os.listdir(date_path):
                    cell_subfolder_path = os.path.join(date_path, cell_subfolder)
                    if not os.path.isdir(cell_subfolder_path):
                        continue

                    # 创建对应的输出文件夹
                    output_cell_subfolder_path = os.path.join(output_date_path, cell_subfolder)
                    os.makedirs(output_cell_subfolder_path, exist_ok=True)

                    print(f"      Processing cell subfolder: {cell_subfolder}")

                    # 获取所有图像文件
                    image_files = sorted([f for f in os.listdir(cell_subfolder_path) if f.endswith('.tiff')])

                    # 限制生成的光流图片数量为16张
                    max_images = 16
                    if len(image_files) < max_images:
                        print(
                            f"        Not enough images ({len(image_files)}) to generate 16 optical flow images. Skipping.")
                        continue

                    # 计算光流并保存
                    count = 0
                    for i in range(len(image_files) - 1):
                        if count >= max_images:
                            break

                        img1_path = os.path.join(cell_subfolder_path, image_files[i])
                        img2_path = os.path.join(cell_subfolder_path, image_files[i + 1])
                        output_filename = os.path.splitext(image_files[i])[0] + '.png'
                        output_path = os.path.join(output_cell_subfolder_path, output_filename)

                        print(f"        Processing image pair: {image_files[i]} -> {image_files[i + 1]}")
                        calculate_optical_flow(img1_path, img2_path, output_path)
                        count += 1


# 主函数
if __name__ == "__main__":
    root_dir = "G:\python\TRN\Speckle"  # 替换为你的数据根目录
    output_root = "G:\python\TRN\optical_flow"  # 输出根目录
    print("Starting optical flow calculation...")
    process_cell_folders(root_dir, output_root)
    print("Optical flow calculation completed.")
