import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConvNet, HyperspectralCNN  # 替换为你的模型文件
from data_generator import HyperspectralDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.__version__)  # 查看PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.get_device_name(0))  # 获取GPU名称

torch.manual_seed(42)
torch.cuda.manual_seed(42)  # 为当前 GPU 设定种子

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 16
epochs = 50
learning_rate = 0.001
num_class = 8

# 加载数据集
root_dir = "Hyper"  # 替换为你的数据集路径
dataset = HyperspectralDataset(root_dir, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model_save_path = "model/best_model.pth"
model = HyperspectralCNN(num_classes=num_class).to(device)
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
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with accuracy: {best_acc:.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}")


# 验证函数
def validate():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Validation - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

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
