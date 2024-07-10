import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import time

from vgg16_model import VGG16
from cnn import ImprovedCNN

base_dir = 'D:\\BaiduNetdiskDownload\\raw-img'
class_names = os.listdir(base_dir)

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Vgg16需要224x224的输入
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历所有文件夹和文件
        for class_index, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('jpg', 'jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图像是RGB模式
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# 模型训练
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        end_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')

        # 在每个epoch结束时评估模型
        print("Evaluating model...")
        evaluate_model(model, test_loader)
        print("Evaluation finished.")


# 模型评估
def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for images, labels in test_loader:
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')


if __name__ == '__main__':
    # 加载完整数据集
    print("Loading dataset...")
    full_dataset = CustomDataset(root_dir=base_dir, transform=transform)
    print(f"Total images: {len(full_dataset)}")

    # 拆分训练集和测试集
    train_size = int(0.8 * len(full_dataset))
    # test_size = int(0.01 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # 创建训练集和测试集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 创建VGG16模型实例
    model = ImprovedCNN()  # 根据类别数量设置输出层
    print("Model initialized.")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 调用训练和测试函数
    print("Starting training...")
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=2)
    print("Training finished. Starting evaluation...")
    evaluate_model(model, test_loader)

    # 保存模型
    # torch.save(model.state_dict(), 'vgg16.pth')
    torch.save(model.state_dict(), 'cnn.pth')
    print("Model saved.")

    # 加载模型
    # model.load_state_dict(torch.load('vgg16.pth'))
