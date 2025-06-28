# MNIST画像認識チュートリアル

PyTorchを使ったMNIST手書き数字認識のチュートリアルです。

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

## モデル構造

- 入力層: 784ユニット（28x28画像）
- 隠れ層: 128ユニット（ReLU活性化）
- 出力層: 10ユニット（0-9の数字分類）