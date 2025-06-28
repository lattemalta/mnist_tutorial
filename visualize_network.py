import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt

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

def visualize_network():
    print("ネットワーク構造を可視化中...")
    
    # モデルとダミー入力を作成
    model = SimpleNet()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # フォワードパスを実行
    output = model(dummy_input)
    
    # 計算グラフを可視化
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('network_structure', cleanup=True)
    
    print("ネットワーク構造図を 'network_structure.png' として保存しました")
    
    # モデルのサマリーも表示
    print("\nモデル構造:")
    print(model)
    
    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")

if __name__ == "__main__":
    visualize_network()