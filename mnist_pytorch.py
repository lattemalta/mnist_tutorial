import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'テスト精度: {accuracy:.2f}%')
    return accuracy

def visualize_predictions(model, test_loader, num_images=6):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        plt.figure(figsize=(12, 8))
        for i in range(num_images):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i][0], cmap='gray')
            plt.title(f'実際: {labels[i].item()}, 予測: {predicted[i].item()}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    print("MNISTデータセットを使った画像認識チュートリアル")
    print("=" * 50)
    
    # データの読み込み
    print("データを読み込み中...")
    train_loader, test_loader = load_data()
    print("データの読み込み完了!")
    
    # モデルの作成
    model = SimpleNet()
    print(f"モデル構造:\n{model}")
    
    # 訓練
    print("\nモデルの訓練を開始...")
    train_model(model, train_loader, epochs=5)
    
    # テスト
    print("\nモデルのテストを実行...")
    accuracy = test_model(model, test_loader)
    
    # 予測の可視化
    print("\n予測結果を可視化...")
    visualize_predictions(model, test_loader)
    
    print(f"\nチュートリアル完了! 最終精度: {accuracy:.2f}%")

if __name__ == "__main__":
    main()