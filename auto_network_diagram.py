import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import OrderedDict

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def get_layer_info(model, input_shape):
    """レイヤーの情報を取得"""
    layer_info = []
    
    def hook_fn(module, input, output):
        # Sequentialやその他のコンテナは除外
        if module.__class__.__name__ in ['Sequential', 'ModuleList', 'ModuleDict']:
            return
            
        if hasattr(output, 'shape'):
            shape = tuple(output.shape[1:])  # バッチ次元を除く
            layer_info.append({
                'name': module.__class__.__name__,
                'input_shape': tuple(input[0].shape[1:]) if input and hasattr(input[0], 'shape') else None,
                'output_shape': shape,
                'params': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
    
    # フックを登録 - 葉ノード（実際の計算を行うレイヤー）のみ
    hooks = []
    for name, module in model.named_modules():
        # コンテナレイヤーは除外し、実際の処理を行うレイヤーのみにフックを登録
        if (len(list(module.children())) == 0 and 
            module.__class__.__name__ not in ['Sequential', 'ModuleList', 'ModuleDict']):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # ダミー入力でフォワードパス
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    # フックを削除
    for hook in hooks:
        hook.remove()
    
    return layer_info

def get_layer_color(layer_name):
    """レイヤータイプに応じた色を返す"""
    color_map = {
        'Linear': '#D4E6F1',
        'Conv2d': '#E8F4FD', 
        'ReLU': '#F8D7DA',
        'Sigmoid': '#F8D7DA',
        'Tanh': '#F8D7DA',
        'Flatten': '#FFE6CC',
        'Dropout': '#F0F0F0',
        'BatchNorm': '#E6F3FF',
        'MaxPool2d': '#FFE6E6',
        'AvgPool2d': '#FFE6E6',
        'default': '#F5F5F5'
    }
    return color_map.get(layer_name, color_map['default'])

def format_shape(shape):
    """形状を文字列に変換"""
    if len(shape) == 1:
        return str(shape[0])
    elif len(shape) == 3:  # C, H, W
        return f"{shape[0]}×{shape[1]}×{shape[2]}"
    else:
        return "×".join(map(str, shape))

def create_auto_network_diagram(model, input_shape, save_name="auto_network_diagram"):
    """モデルから自動的にネットワーク図を生成"""
    
    # レイヤー情報を取得
    layers = get_layer_info(model, input_shape)
    
    # 入力レイヤーを追加
    input_layer = {
        'name': 'Input',
        'input_shape': None,
        'output_shape': input_shape,
        'params': 0
    }
    layers.insert(0, input_layer)
    
    # 図のサイズを計算
    num_layers = len(layers)
    fig_width = max(12, num_layers * 2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))
    
    # レイヤー間の間隔
    layer_spacing = fig_width / (num_layers + 1)
    
    # 各レイヤーを描画
    total_params = 0
    for i, layer in enumerate(layers):
        x_pos = (i + 1) * layer_spacing
        
        # レイヤーのサイズを出力形状に基づいて決定
        if layer['output_shape']:
            output_size = np.prod(layer['output_shape'])
            # サイズを対数スケールで正規化
            height = max(1.5, min(4, np.log10(output_size + 1) * 0.8))
        else:
            height = 2
        
        width = 1.2
        
        # 長方形を描画
        color = get_layer_color(layer['name'])
        rect = patches.Rectangle(
            (x_pos - width/2, 2 - height/2),
            width, height,
            linewidth=2, edgecolor='black', facecolor=color
        )
        ax.add_patch(rect)
        
        # レイヤー名と出力形状を表示
        if layer['output_shape']:
            shape_str = format_shape(layer['output_shape'])
            label = f"{layer['name']}\n{shape_str}"
        else:
            label = layer['name']
        
        ax.text(x_pos, 2, label, 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # パラメータ数を表示
        if layer['params'] > 0:
            param_str = f"{layer['params']:,} params" if layer['params'] < 1000 else f"{layer['params']/1000:.1f}K params"
            ax.text(x_pos, 0.3, param_str, 
                    ha='center', va='center', fontsize=7, style='italic')
            total_params += layer['params']
        
        # 矢印を描画（最後のレイヤー以外）
        if i < len(layers) - 1:
            arrow_start = x_pos + width/2
            arrow_end = (i + 2) * layer_spacing - width/2
            ax.annotate('', xy=(arrow_end, 2), xytext=(arrow_start, 2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # タイトル
    model_name = model.__class__.__name__
    ax.text(fig_width/2, 5.5, f'{model_name} Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 総パラメータ数
    ax.text(fig_width/2, -1.5, f'Total Parameters: {total_params:,}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # 軸の設定
    ax.set_xlim(0, fig_width)
    ax.set_ylim(-3, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 保存
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.svg', bbox_inches='tight')
    print(f"ネットワーク図を '{save_name}.png' と '{save_name}.svg' として保存しました")
    
    return fig, ax

# SimpleNetクラスの定義
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

# より複雑なモデルの例
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

if __name__ == "__main__":
    # SimpleNetのテスト
    print("SimpleNet の図を生成中...")
    model = SimpleNet()
    create_auto_network_diagram(model, (1, 28, 28), "simple_net_diagram")
    
    # CNNNetのテスト
    print("\nCNNNet の図を生成中...")
    cnn_model = CNNNet()
    create_auto_network_diagram(cnn_model, (1, 28, 28), "cnn_net_diagram")
    
    plt.show()