import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConvNet, HyperspectralCNN, TwoStreamConvNet  # 替换为你的模型文件
from data_generator import HyperspectralDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from datetime import datetime

print(torch.__version__)  # 查看PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.get_device_name(0))  # 获取GPU名称

torch.manual_seed(42)
torch.cuda.manual_seed(42)  # 为当前 GPU 设定种子

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 32
epochs = 100
learning_rate = 0.001
num_class = 8

transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True)
])

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")


# 加载数据集
Hyper_dir = "Data/Hyper"  # 替换为你的数据集路径
optical_flow_dir = "Data/Optical_flow"

dataset = HyperspectralDataset(Hyper_dir,optical_flow_dir, transform=transform)
train_size = int(0.6 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


model = TwoStreamConvNet(num_classes=num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练函数
def train():
    best_acc = 0.0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for hyperspectral_image, optical_image, labels in progress_bar:
            hyperspectral_image, optical_image, labels = hyperspectral_image.to(device), optical_image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(hyperspectral_image,optical_image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(loss=running_loss / len(train_loader))

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            model_save_path = f"model/best_model_bs{batch_size}_ep{epochs}_lr{learning_rate}_nc{num_class}_{current_datetime}_best_acc{best_acc}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with accuracy: {best_acc:.4f} \n")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {acc:.4f} \n")

        print("Testing...\n")
        model.eval()
        test_all_preds, test_all_labels = [], []
        with torch.no_grad():
            test_progress_bar = tqdm(test_loader, desc="Testing")
            for hyperspectral_image, optical_image, labels in test_progress_bar:
                hyperspectral_image, optical_image, labels = hyperspectral_image.to(device), optical_image.to(
                    device), labels.to(device)
                outputs = model(hyperspectral_image, optical_image)
                _, preds = torch.max(outputs, 1)
                test_all_preds.extend(preds.cpu().numpy())
                test_all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(test_all_labels, test_all_preds)
        print(f"Test - Accuracy: {acc:.4f}\n")

# 验证函数
def validate():
    print("Validation...\n")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Validing")
        for hyperspectral_image, optical_image, labels in test_progress_bar:
            hyperspectral_image, optical_image, labels = hyperspectral_image.to(device), optical_image.to(device), labels.to(device)
            outputs = model(hyperspectral_image, optical_image)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Validation - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} \n")

    # 计算并绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(8), yticklabels=range(8))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    train()
    validate()
