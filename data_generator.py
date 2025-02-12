import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from PIL import Image


class HyperspectralDataset(Dataset):
    def __init__(self, Hyper_dir, optical_flow_dir, transform=None):
        self.Hyper_dir = Hyper_dir  # 高光谱图像的根目录
        self.optical_flow_dir = optical_flow_dir  # 光流图像的根目录
        self.transform = transform
        self.hyper_image_path = []
        self.optical_flow_paths = []
        self.labels = []

        # 细胞类型和标签的映射
        label_map = {
            "Jurkat_Mosaic_alive": 0, "Jurkat_Mosaic_dead": 1,
            "K562_Mosaic_alive": 2, "K562_Mosaic_dead": 3,
            "MEC1_Mosaic_alive": 4, "MEC1_Mosaic_dead": 5,
            "THP1_Mosaic_alive": 6, "THP1_Mosaic_dead": 7
        }

        # 遍历文件夹结构，找到所有高光谱图像
        for cell_type in ["Jurkat", "K562", "MEC1", "THP1"]:
            for state in ["_Mosaic_alive", "_Mosaic_dead"]:
                folder_name = cell_type + state
                folder_path = os.path.join(Hyper_dir, cell_type, folder_name)
                optical_flow_folder_path = os.path.join(optical_flow_dir, cell_type, folder_name)
                for sample_folder in os.listdir(folder_path):
                    sample_path = os.path.join(folder_path, sample_folder)
                    optical_flow_sample_path = os.path.join(optical_flow_folder_path, sample_folder)
                    if os.path.isdir(sample_path) and os.path.isdir(optical_flow_sample_path):
                        channel_images = [os.path.join(sample_path, f) for f in sorted(os.listdir(sample_path))]
                        optical_flow_images = [os.path.join(optical_flow_sample_path, f) for f in sorted(os.listdir(optical_flow_sample_path))]
                        if len(channel_images) == 16 and len(optical_flow_images) == 16:  # 确保完整的高光谱图像和光流图像
                            self.hyper_image_path.append(channel_images)  # 修改变量名
                            self.optical_flow_paths.append(optical_flow_images)
                            self.labels.append(label_map[folder_name])  #

    def __len__(self):
        return len(self.hyper_image_path)

    def __getitem__(self, idx):
        channel_images = self.hyper_image_path[idx]  # 修改变量名
        hyperspectral_image = np.stack([np.array(Image.open(img).convert("L"), dtype=np.float32) for img in channel_images], axis=0)  # 形状 (16, H, W)
        hyperspectral_image = torch.tensor(hyperspectral_image) / 255.0  # 归一化

        # 加载光流图像
        optical_flow_images = self.optical_flow_paths[idx]
        optical_image = np.stack([np.array(Image.open(img).convert("L"), dtype=np.float32) for img in optical_flow_images], axis=0)  # 形状 (16, H, W)
        optical_image = torch.tensor(optical_image) / 255.0  # 归一化

        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 转换为 PyTorch 张量

        # 应用变换（如果有）
        if self.transform:
            hyperspectral_image = self.transform(hyperspectral_image)
            optical_image = self.transform(optical_image)

        return hyperspectral_image, optical_image, label

if __name__ == '__main__':
    Hyper_dir = "Data/Hyper"  # 替换为你的数据集路径
    optical_flow_dir = "Data/Optical_flow"
    dataset = HyperspectralDataset(Hyper_dir, optical_flow_dir, transform=None)
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))  # 80% 作为训练集
    val_size = len(dataset) - train_size  # 20% 作为验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  # Windows 请设置 num_workers=0
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 取一个 batch 进行测试
    for hyperspectral_image, optical_image, labels in train_dataloader:
        print("Train Batch Hyperspectral Image Shape:", hyperspectral_image.shape)  # (batch_size, 16, H, W)
        print("Train Batch Optical Image Shape:", optical_image.shape)  # (batch_size, 16, H, W)
        print("Train Batch Labels:", labels)  # (batch_size,)
        break  # 只打印一个 batch

    for hyperspectral_image, optical_image, labels in val_dataloader:
        print("Train Batch Hyperspectral Image Shape:", hyperspectral_image.shape)  # (batch_size, 16, H, W)
        print("Train Batch Optical Image Shape:", optical_image.shape)  # (batch_size, 16, H, W)
        print("Train Batch Labels:", labels)  # (batch_size,)
        break  # 只打印一个 batch

