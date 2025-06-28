import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（警告を避けるため）
plt.rcParams['font.family'] = ['DejaVu Sans']

def generate_mnist_sample_images():
    """MNISTデータセットからサンプル画像を生成"""
    
    # データセットを読み込み
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform
    )
    
    # 各数字（0-9）から1つずつサンプルを取得
    samples = {}
    for i, (image, label) in enumerate(train_dataset):
        label_int = label.item() if hasattr(label, 'item') else label
        if label_int not in samples:
            samples[label_int] = image
        
        # 全ての数字のサンプルが集まったら終了
        if len(samples) == 10:
            break
    
    # 2x5のグリッドでサンプル画像を表示
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle('MNIST Hand-written Digit Samples', fontsize=16, fontweight='bold')
    
    for i in range(10):
        row = i // 5
        col = i % 5
        
        # 画像を表示
        image = samples[i].squeeze().numpy()
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'Label: {i}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('../mnist_samples.png', dpi=300, bbox_inches='tight')
    print("MNIST sample images saved as 'mnist_samples.png'")
    
    # 個別の数字の詳細表示用の画像も生成
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('MNIST Image Details (28x28 pixels)', fontsize=14, fontweight='bold')
    
    # 数字7の詳細表示
    sample_7 = samples[7].squeeze().numpy()
    
    # オリジナル画像
    axes[0].imshow(sample_7, cmap='gray')
    axes[0].set_title('Original (28x28)')
    axes[0].axis('off')
    
    # ピクセル値を可視化
    axes[1].imshow(sample_7, cmap='viridis')
    axes[1].set_title('Pixel Values')
    axes[1].axis('off')
    
    # 3D表示風
    x = np.arange(0, 28)
    y = np.arange(0, 28)
    X, Y = np.meshgrid(x, y)
    
    # 2Dコンター表示
    contour = axes[2].contour(X, Y, sample_7, levels=10, cmap='gray')
    axes[2].set_title('Contour Lines')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../mnist_details.png', dpi=300, bbox_inches='tight')
    print("MNIST detail images saved as 'mnist_details.png'")

if __name__ == "__main__":
    generate_mnist_sample_images()