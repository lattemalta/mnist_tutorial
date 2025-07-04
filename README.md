# MNIST画像認識チュートリアル

PyTorchを使ったMNIST手書き数字認識のチュートリアルです。

## MNISTデータセットについて

MNISTは機械学習の入門によく使われる手書き数字のデータセットです。

![MNIST Samples](mnist_samples.png)

- **データ数**: 訓練用60,000枚、テスト用10,000枚
- **画像サイズ**: 28×28ピクセル（グレースケール）
- **クラス数**: 10クラス（0-9の数字）
- **用途**: 手書き数字認識（画像分類タスク）

![MNIST Details](mnist_details.png)

このデータセットは、ディープラーニングの基本概念を学ぶのに最適で、比較的小さなモデルでも高い精度を達成できます。

## ファイル説明

- `mnist_pytorch.py`: 完全版（可視化機能付き）
- `simple_mnist.py`: 簡易版（基本的な学習と評価）

## 必要なライブラリ

```bash
pip3 install torch torchvision matplotlib numpy
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

## 結果

簡易版で1エポック訓練後、テスト精度約90%を達成しました。

### モデル性能比較

5エポック訓練後の詳細な比較結果：

![Model Comparison](model_comparison.png)

![Final Comparison](final_comparison.png)

**最終結果（5エポック後）:**
- **SimpleNet**: 訓練精度 98.4%, テスト精度 97.9%
- **CNNNet**: 訓練精度 98.7%, テスト精度 99.1%

CNNNetの方がより高い精度を達成し、特にテスト精度で優秀な結果を示しています。

## モデル構造

### SimpleNet（基本的な全結合ニューラルネットワーク）

```mermaid
flowchart LR
    A(Input<br/>28×28<br/>784 nodes) --> B(Flatten<br/>784 nodes)
    B --> C(Linear<br/>128 nodes<br/>100,480 params)
    C --> D(ReLU)
    D --> E(Linear<br/>128 nodes<br/>16,512 params)
    E --> F(ReLU)
    F --> G(Linear<br/>10 nodes<br/>1,290 params)
    G --> H(Output<br/>10 classes)
    
    classDef input fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#FFFFFF
    classDef linear fill:#7B68EE,stroke:#5A4FCF,stroke-width:2px,color:#FFFFFF
    classDef activation fill:#FF6B6B,stroke:#E55555,stroke-width:2px,color:#FFFFFF
    classDef flatten fill:#FFB74D,stroke:#F57C00,stroke-width:2px,color:#FFFFFF
    classDef output fill:#66BB6A,stroke:#4CAF50,stroke-width:2px,color:#FFFFFF
    
    class A input
    class B flatten
    class C,E,G linear
    class D,F activation
    class H output
```

- 総パラメータ数: 118,282

### CNNNet（畳み込みニューラルネットワーク）

```mermaid
flowchart LR
    A(Input<br/>1×28×28) --> B(Conv2d<br/>1→32<br/>320 params)
    B --> C(ReLU)
    C --> D(MaxPool2d<br/>2×2)
    D --> E(Conv2d<br/>32→64<br/>18,496 params)
    E --> F(ReLU)
    F --> G(MaxPool2d<br/>2×2)
    G --> H(Flatten<br/>3136 nodes)
    H --> I(Linear<br/>128 nodes<br/>401,536 params)
    I --> J(ReLU)
    J --> K(Dropout<br/>0.5)
    K --> L(Linear<br/>10 nodes<br/>1,290 params)
    L --> M(Output<br/>10 classes)
    
    classDef input fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#FFFFFF
    classDef conv fill:#26A69A,stroke:#00695C,stroke-width:2px,color:#FFFFFF
    classDef pool fill:#FF7043,stroke:#D84315,stroke-width:2px,color:#FFFFFF
    classDef linear fill:#7B68EE,stroke:#5A4FCF,stroke-width:2px,color:#FFFFFF
    classDef activation fill:#FF6B6B,stroke:#E55555,stroke-width:2px,color:#FFFFFF
    classDef flatten fill:#FFB74D,stroke:#F57C00,stroke-width:2px,color:#FFFFFF
    classDef dropout fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#FFFFFF
    classDef output fill:#66BB6A,stroke:#4CAF50,stroke-width:2px,color:#FFFFFF
    
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

