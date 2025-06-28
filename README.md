# MNIST画像認識チュートリアル

PyTorchを使ったMNIST手書き数字認識のチュートリアルです。

## ファイル説明

- `mnist_pytorch.py`: 完全版（可視化機能付き）
- `simple_mnist.py`: 簡易版（基本的な学習と評価）
- `auto_network_diagram.py`: ネットワーク構造自動可視化ツール
- `visualize_network.py`: torchvizを使った詳細計算グラフ可視化

## 必要なライブラリ

```bash
pip3 install torch torchvision matplotlib numpy torchviz graphviz
```

macOSでの追加セットアップ：
```bash
brew install graphviz
```

## 実行方法

### 簡易版
```bash
python3 simple_mnist.py
```

### 完全版（可視化付き）
```bash
python3 mnist_pytorch.py
```

### ネットワーク構造可視化
```bash
python3 auto_network_diagram.py
```

## 結果

簡易版で1エポック訓練後、テスト精度約90%を達成しました。

## モデル構造

### SimpleNet（基本的な全結合ニューラルネットワーク）

```mermaid
graph LR
    A[Input<br/>28×28<br/>784 nodes] --> B[Flatten<br/>784 nodes]
    B --> C[Linear<br/>128 nodes<br/>100,480 params]
    C --> D[ReLU]
    D --> E[Linear<br/>128 nodes<br/>16,512 params]
    E --> F[ReLU]
    F --> G[Linear<br/>10 nodes<br/>1,290 params]
    G --> H[Output<br/>10 classes]
    
    classDef input fill:#E8F4FD
    classDef linear fill:#D4E6F1
    classDef activation fill:#F8D7DA
    classDef flatten fill:#FFE6CC
    classDef output fill:#D5F4E6
    
    class A input
    class B flatten
    class C,E,G linear
    class D,F activation
    class H output
```

- 総パラメータ数: 118,282

### CNNNet（畳み込みニューラルネットワーク）

```mermaid
graph LR
    A[Input<br/>1×28×28] --> B[Conv2d<br/>1→32<br/>320 params]
    B --> C[ReLU]
    C --> D[MaxPool2d<br/>2×2]
    D --> E[Conv2d<br/>32→64<br/>18,496 params]
    E --> F[ReLU]
    F --> G[MaxPool2d<br/>2×2]
    G --> H[Flatten<br/>3136 nodes]
    H --> I[Linear<br/>128 nodes<br/>401,536 params]
    I --> J[ReLU]
    J --> K[Dropout<br/>0.5]
    K --> L[Linear<br/>10 nodes<br/>1,290 params]
    L --> M[Output<br/>10 classes]
    
    classDef input fill:#E8F4FD
    classDef conv fill:#B8E6B8
    classDef pool fill:#FFE6E6
    classDef linear fill:#D4E6F1
    classDef activation fill:#F8D7DA
    classDef flatten fill:#FFE6CC
    classDef dropout fill:#F0F0F0
    classDef output fill:#D5F4E6
    
    class A input
    class B,E conv
    class C,F,J activation
    class D,G pool
    class H flatten
    class I,L linear
    class K dropout
    class M output
```

- 総パラメータ数: 421,642

## ネットワーク構造可視化機能

このプロジェクトには、任意のPyTorchモデルから自動的にネットワーク構造図を生成する汎用的な機能が含まれています：

- **自動レイヤー検出**: モデルの構造を自動解析
- **レイヤータイプ別色分け**: Conv2d、Linear、ReLUなど種類ごとに色分け
- **パラメータ数表示**: 各レイヤーと総パラメータ数を表示
- **複数形式対応**: PNG/SVG形式で出力

使用例：
```python
from auto_network_diagram import create_auto_network_diagram

# 任意のPyTorchモデルで使用可能
model = YourModel()
create_auto_network_diagram(model, input_shape=(1, 28, 28), save_name="your_model")
```