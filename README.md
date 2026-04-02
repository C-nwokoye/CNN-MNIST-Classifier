# 🔢 CNN Digit Classifier (MNIST)

A Convolutional Neural Network (CNN) built in PyTorch that classifies handwritten digits (0–9) from the MNIST dataset, achieving **97.52% test accuracy** after 500 training epochs — outperforming both MLP (91.00%) and SLP (89.54%) baselines.

---

## 📊 Results

### Accuracy Comparison (500 Epochs)

| Model | Train Loss | Test Loss | Test Accuracy |
|-------|-----------|-----------|---------------|
| **CNN** | **0.0805** | **0.00106** | **97.52%** |
| MLP | 0.3130 | 0.00469 | 91.00% |
| SLP | 0.399 | 0.00590 | 89.54% |

### CPU vs GPU Training (40 Epochs)

| | CPU | GPU |
|---|---|---|
| Train Time | 24 min 46 sec | 4 min 45 sec |
| Test Accuracy | 87.64% | 87.70% |

> The GPU achieves the same accuracy ~5x faster than CPU training.

---

## 🧠 Model Architecture

```
MNISTConvNet
├── Conv Block 1:  Conv2d(1 → 32, 5×5, same padding) → ReLU → MaxPool2d(2)
├── Conv Block 2:  Conv2d(32 → 64, 5×5, same padding) → ReLU → MaxPool2d(2)
└── FC Head:       Flatten → Linear(3136 → 1024) → Dropout(0.5) → Linear(1024 → 10)
```

**Training configuration:**
- Optimizer: SGD
- Loss: CrossEntropyLoss
- Learning rate: 1e-4
- Batch size: 64
- Epochs: 500
- Seed: 0 (for reproducibility)

---

## 📁 Project Structure

```
cnn-digit-classifier/
├── cnn.py          # CNN architecture (MNISTConvNet)
├── cnn_train.py    # Training loop, saves model weights to .pt file
├── cnn_test.py     # Evaluation, confusion matrix generation
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy
```

### Train the Model

```bash
python cnn_train.py
```

Trains for 500 epochs and saves weights to `cnn_mnist_500.pt`. The MNIST dataset is downloaded automatically on first run.

### Evaluate the Model

```bash
python cnn_test.py
```

Loads saved weights and prints test loss, test accuracy, and a full confusion matrix.

---

## 🔍 Notable Findings

- The CNN was the **only model** to correctly classify more than one digit class over 1000 times
- All models found **1s easiest** to classify and **5s hardest** (most often confused with 3s)
- Increasing from 40 → 500 epochs gave the CNN a **+9.8% accuracy boost**, while MLP and SLP saw minimal gains
- CrossEntropyLoss heavily rewards the CNN for the final 90% → 97% improvement, as reflected in its significantly lower test loss (0.00106 vs 0.00469 for MLP)

---

## 📚 Tech Stack

- **Python 3.x**
- **PyTorch**
- **torchvision**
- **NumPy**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
