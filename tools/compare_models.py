import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans']

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

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def load_data():
    """データローダーを作成"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_and_evaluate(model, model_name, train_loader, test_loader, epochs=10):
    """モデルを訓練し、エポックごとの精度を記録"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n=== Training {model_name} ===")
    
    for epoch in range(epochs):
        # 訓練
        model.train()
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # テスト
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    return train_accuracies, test_accuracies

def compare_models():
    """両方のモデルを比較"""
    print("Loading data...")
    train_loader, test_loader = load_data()
    
    # モデルを初期化
    simple_net = SimpleNet()
    cnn_net = CNNNet()
    
    epochs = 5
    
    # SimpleNetを訓練
    simple_train_acc, simple_test_acc = train_and_evaluate(
        simple_net, "SimpleNet", train_loader, test_loader, epochs
    )
    
    # CNNNetを訓練
    cnn_train_acc, cnn_test_acc = train_and_evaluate(
        cnn_net, "CNNNet", train_loader, test_loader, epochs
    )
    
    # グラフを作成
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 訓練精度の比較
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, simple_train_acc, 'b-o', label='SimpleNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, cnn_train_acc, 'r-s', label='CNNNet', linewidth=2, markersize=6)
    plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # テスト精度の比較
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, simple_test_acc, 'b-o', label='SimpleNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, cnn_test_acc, 'r-s', label='CNNNet', linewidth=2, markersize=6)
    plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('../model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison graph saved as 'model_comparison.png'")
    
    # 最終結果の比較グラフも作成
    plt.figure(figsize=(10, 6))
    
    models = ['SimpleNet', 'CNNNet']
    final_train = [simple_train_acc[-1], cnn_train_acc[-1]]
    final_test = [simple_test_acc[-1], cnn_test_acc[-1]]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, final_train, width, label='Training Accuracy', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, final_test, width, label='Test Accuracy', color='lightcoral', alpha=0.8)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Final Accuracy Comparison (After 5 Epochs)', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 100)
    
    # 数値をバーの上に表示
    for i, (train, test) in enumerate(zip(final_train, final_test)):
        plt.text(i - width/2, train + 1, f'{train:.1f}%', ha='center', fontweight='bold')
        plt.text(i + width/2, test + 1, f'{test:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../final_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Final comparison graph saved as 'final_comparison.png'")
    
    # 結果をテキストファイルにも保存
    with open('../comparison_results.txt', 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"SimpleNet - Final Train Accuracy: {simple_train_acc[-1]:.2f}%\n")
        f.write(f"SimpleNet - Final Test Accuracy: {simple_test_acc[-1]:.2f}%\n\n")
        f.write(f"CNNNet - Final Train Accuracy: {cnn_train_acc[-1]:.2f}%\n")
        f.write(f"CNNNet - Final Test Accuracy: {cnn_test_acc[-1]:.2f}%\n\n")
        f.write("Epoch-by-epoch results:\n\n")
        f.write("SimpleNet:\n")
        for i, (train, test) in enumerate(zip(simple_train_acc, simple_test_acc)):
            f.write(f"Epoch {i+1}: Train {train:.2f}%, Test {test:.2f}%\n")
        f.write("\nCNNNet:\n")
        for i, (train, test) in enumerate(zip(cnn_train_acc, cnn_test_acc)):
            f.write(f"Epoch {i+1}: Train {train:.2f}%, Test {test:.2f}%\n")
    
    print("Results saved to 'comparison_results.txt'")

if __name__ == "__main__":
    compare_models()