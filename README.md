# 🧠 DL-Framework: A Simple Deep Learning Library in NumPy

A minimal deep learning framework built **from scratch** using **only NumPy** — no external ML libraries required. Ideal for learning, experimentation, and understanding how neural networks work under the hood.

---

## 🚀 Features

- Build custom **feedforward neural networks** for:
  - **Classification**
  - **Regression**
- Modular `Layer` class for easy network construction
- Fully working **forward and backward propagation**
- Loss functions:
  - Mean Squared Error (**MSE**)
  - **Cross-Entropy**
- Activation functions:
  - **ReLU**
  - **Softmax**

---

## 🧱 Architecture Overview

### ✅ Layer Class
Each layer handles:
- Weight initialization
- Activation computation
- Gradient propagation

Layers can be stacked to form a full network with customizable architecture.
