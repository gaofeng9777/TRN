import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
from PIL import Image


class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
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
                folder_path = os.path.join(root_dir, cell_type, cell_type + state)
                for sample_folder in os.listdir(folder_path):
                    sample_path = os.path.join(folder_path, sample_folder)
                    if os.path.isdir(sample_path):
                        channel_images = [os.path.join(sample_path, f) for f in sorted(os.listdir(sample_path))]
                        if len(channel_images) == 16:  # 确保完整的高光谱图像
                            self.image_paths.append(channel_images)
                            self.labels.append(label_map[folder_name])  # 添加标签

    def __len__(self):
        #print(len(self.image_paths))
        return len(self.image_paths)

    def __getitem__(self, idx):
        channel_images = self.image_paths[idx]
        images = [np.array(Image.open(img).convert("L"), dtype=np.float32) for img in channel_images]  # 转换为灰度图
        hyperspectral_image = np.stack(images, axis=0)  # 形状 (16, H, W)
        hyperspectral_image = torch.tensor(hyperspectral_image) / 255.0  # 归一化

        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 转换为 PyTorch 张量

        if self.transform:
            hyperspectral_image = self.transform(hyperspectral_image)

        return hyperspectral_image,label

if __name__ == '__main__':
    root_dir = "Hyper"  # 替换为你的数据集路径
    dataset = HyperspectralDataset(root_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))  # 80% 作为训练集
    val_size = len(dataset) - train_size  # 20% 作为验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  # Windows 请设置 num_workers=0
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 取一个 batch 进行测试
    for images, labels in train_dataloader:
        print("Train Batch Image Shape:", images.shape)  # (batch_size, 16, H, W)
        print("Train Batch Labels:", labels)  # (batch_size,)
        break  # 只打印一个 batch

    for images, labels in val_dataloader:
        print("Validation Batch Image Shape:", images.shape)  # (batch_size, 16, H, W)
        print("Validation Batch Labels:", labels)  # (batch_size,)
        break  # 只打印一个 batch


