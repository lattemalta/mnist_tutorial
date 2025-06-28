import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():
    print("MNIST画像認識チュートリアル（簡易版）")
    print("=" * 40)
    
    # データの読み込み
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("MNISTデータセットをダウンロード中...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)
    
    print("データの読み込み完了!")
    print(f"訓練データ: {len(train_dataset)} 画像")
    print(f"テストデータ: {len(test_dataset)} 画像")
    
    # モデルの作成
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nモデル構造:")
    print(model)
    
    # 簡単な訓練（1エポックのみ）
    print("\n訓練開始（1エポック）...")
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if i >= 100:  # 最初の100バッチのみ
            break
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 20 == 19:
            print(f"バッチ {i+1}/100, 損失: {total_loss/20:.4f}")
            total_loss = 0
    
    # テスト
    print("\nテスト実行...")
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
    print(f"テスト精度: {accuracy:.2f}%")
    print("\nチュートリアル完了!")

if __name__ == "__main__":
    main()