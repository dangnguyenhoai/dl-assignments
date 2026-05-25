# Weekly Progress Report

* **Author:** Dang Nguyen Hoai - 2001230257
* **Topic:** Lab 03 - Artificial Neural Networks (ANN) for Image Classification
* **Date:** Week 03 - May 2026

---

## 1. Exploratory Data Analysis & Preprocessing

This laboratory assignment involves building and training Multi-Layer Perceptrons (MLPs) on two benchmark image classification datasets: **Fashion MNIST** and **CIFAR-10**.

### A. Fashion MNIST Dataset
* **Dataset Characteristics:** Contains 70,000 grayscale images (60,000 training, 10,000 test) of size $28 \times 28$ pixels across 10 fashion categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
* **Preprocessing Pipeline:**
  1. **Dimensionality Reduction:** Grayscale 2D image matrices of shape $(28, 28)$ are flattened into 1D feature vectors of size $784$ ($28 \times 28 = 784$).
  2. **Min-Max Scaling:** Pixels are normalized to the range $[0, 1]$ by dividing by $255.0$ and then fit-transformed using `MinMaxScaler` to ensure numerical stability and centered inputs:
     $$X' = \frac{X - \min(X)}{\max(X) - \min(X)}$$

### B. CIFAR-10 Dataset (`ex_1.ipynb`)
* **Dataset Characteristics:** Contains 60,000 color images (50,000 training, 10,000 validation) of size $32 \times 32 \times 3$ (RGB) across 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
* **Preprocessing Pipeline:**
  1. **Scaling:** Normalized color channels by dividing by $255.0$ to scale all inputs into the interval $[0.0, 1.0]$.
  2. **Target Reshaping:** Target label shapes are reshaped from $(N, 1)$ to a 1D vector of shape $(N,)$ to properly align with Keras's standard categorical APIs.

---

## 2. Model Architectures

We designed two feedforward Artificial Neural Network (ANN) topologies for image classification tasks using Keras:

### A. Fashion MNIST MLP Architecture (`lab_03.ipynb`)
* **Input Layer:** $784$ nodes (representing the flattened grayscale pixels).
* **Hidden Layer:** $256$ neurons with **ReLU** (Rectified Linear Unit) activation, initialized using `he_normal`.
* **Output Layer:** $10$ neurons with **Softmax** activation (producing a class probability distribution), initialized using `he_normal`.
* **Loss Function:** Sparse Categorical Crossentropy, which computes cross-entropy loss directly from integer targets without requiring one-hot encoding:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
* **Optimizer:** Adam (Adaptive Moment Estimation) with a default learning rate of $0.001$, combining the advantages of AdaGrad and RMSProp.

### B. CIFAR-10 MLP Architecture (`ex_1.ipynb`)
For the color image dataset, a much deeper feedforward topology was utilized to capture spatial patterns in $3$ color channels:
* **Input Layer:** `Flatten` layer converting $32 \times 32 \times 3$ matrices into a $3072$-dimensional input vector.
* **First Hidden Layer:** $1024$ nodes, **ReLU** activation, `he_normal` initialization, L2 regularization ($\lambda = 0.0001$), followed by **Batch Normalization** and **Dropout** ($30\%$).
* **Second Hidden Layer:** $512$ nodes, **ReLU** activation, `he_normal` initialization, L2 regularization ($\lambda = 0.0001$), followed by **Batch Normalization** and **Dropout** ($30\%$).
* **Third Hidden Layer:** $256$ nodes, **ReLU** activation, `he_normal` initialization, L2 regularization ($\lambda = 0.0001$), followed by **Batch Normalization** and **Dropout** ($25\%$).
* **Output Layer:** $10$ nodes with **Softmax** activation.
* **Compilation Details:**
  * **Optimizer:** Adam with learning rate $\eta = 0.001$.
  * **Loss Function:** Sparse Categorical Crossentropy.
  * **Callbacks:** **EarlyStopping** (monitoring validation loss with `patience=8` and restoring best weights) and **ReduceLROnPlateau** (reducing learning rate by a factor of $0.5$ if validation loss plateaus for $3$ epochs).

---

## 3. Deep Theoretical Analysis: Why `he_normal` Initialization?

The choice of weight initialization strategy is one of the most critical hyperparameters in Deep Learning. For both models, **He Normal (Kaiming Normal)** was chosen over standard Uniform or Xavier (Glorot) initialization. Below is the mathematical and practical rationale behind this choice.

### A. The Variance Scaling Problem
In a deep network, if weights are initialized too small, the variance of the input signals will exponentially shrink as they propagate forward through the layers. By the time the forward pass reaches the final layers, the activations will have collapsed near zero (vanishing activations). Conversely, if the weights are initialized too large, the variance will exponentially explode, leading to numerical overflow (exploding activations).
Similarly, during the backward pass, gradients will vanish or explode, making training impossible. To prevent this, the variance of the activations of each layer should remain constant:
$$\text{Var}(z^{[l]}) = \text{Var}(z^{[l-1]})$$

### B. The Drawback of Random Uniform Initialization
A standard random uniform initializer (e.g., $W \sim \mathcal{U}(-r, r)$ with a fixed $r = 0.05$) does not scale its bounds based on the size of the layer (the number of input connections, or $n_{\text{in}}$). 
Under a uniform distribution, the variance of a single weight is:
$$\text{Var}(W) = \frac{(r - (-r))^2}{12} = \frac{r^2}{3}$$
If a layer has $n_{\text{in}}$ inputs (e.g., $784$ in Fashion MNIST or $3072$ in CIFAR-10), the variance of the output of that layer (assuming inputs are independent and standardized) is:
$$\text{Var}(z) \approx n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)$$
With a fixed $r$, the output variance scales linearly with $n_{\text{in}}$. For wide layers, this leads to extremely large activation magnitudes, causing gradients to explode or saturation to occur.

### C. Xavier (Glorot) vs. He (Kaiming) Initialization
* **Xavier (Glorot) Initialization** was designed under the assumption that the activation functions are linear or symmetric (like sigmoid or hyperbolic tangent). Under this assumption, the optimal variance to keep signal flow stable is:
  $$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \quad \text{or} \quad \text{Var}(W) = \frac{1}{n_{\text{in}}}$$
* **The ReLU Problem:** The ReLU activation function $f(x) = \max(0, x)$ is asymmetric and non-linear. Crucially, it sets all negative values to zero. On average, a random input vector will have half of its elements zeroed out by ReLU. Consequently:
  $$E[f(x)^2] \approx \frac{1}{2} E[x^2]$$
  This means that passing a signal through a ReLU layer cuts its variance in half. If we use Xavier initialization in a deep network with ReLU activations, the variance of activations will halve at each layer:
  $$\text{Var}(a^{[l]}) \approx \left(\frac{1}{2}\right)^l \text{Var}(x)$$
  In a deep network, this leads to a rapid exponential decay of activation values, making training extremely slow or halting it entirely (vanishing gradients).

* **He Normal (`he_normal`) Solution:** Kaiming He et al. (2015) introduced a correction factor of $2$ to scale the initialization variance to compensate for the half-rectification of ReLU:
  $$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$
  Weights are drawn from a normal distribution centered at zero with a standard deviation of:
  $$\sigma = \sqrt{\frac{2}{n_{\text{in}}}}$$
  
### D. Summary Comparison

| Initializer | Formula / Bounds | Designed For | Behavior with ReLU |
| :--- | :--- | :--- | :--- |
| **Uniform (Fixed)** | $W \sim \mathcal{U}(-r, r)$ | Shallow networks, no scaling | Activations blow up ($n_{\text{in}}$ large) or vanish ($n_{\text{in}}$ small) |
| **Xavier (Glorot)** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Sigmoid / Tanh | Activation variance decays exponentially by a factor of $0.5^l$ |
| **He (Kaiming) Normal** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ | ReLU / Leaky ReLU | **Maintains constant variance across deep layers (Optimal)** |

---

## 4. Training & Model Performance

### A. Fashion MNIST Results
The model was trained for 20 epochs in the initial exploratory stage and evaluated on validation datasets.

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.4881 | 82.70% | 0.3774 | 86.43% |
| 5 | 0.2836 | 89.40% | 0.3384 | 88.38% |
| 10 | 0.2257 | 91.53% | 0.3101 | 89.28% |
| 15 | 0.1901 | 92.83% | 0.3432 | 88.50% |
| 20 | 0.1612 | 93.86% | 0.3883 | 88.42% |

* **Analysis:** The model converges quickly. The validation loss reaches its minimum around Epoch 8 ($\text{Loss} \approx 0.3040$) with a validation accuracy peaking at $\approx 89.48\%$. Beyond this point, training loss continues to drop smoothly (reaching $0.1612$ at Epoch 20) while validation loss oscillates slightly upwards, representing standard, mild overfitting as the MLP memorizes specific high-frequency details.
* **Accuracy & Loss Curves:**
  
  ![Fashion MNIST Accuracy](result/lab_03_model_accuracy.png)
  ![Fashion MNIST Loss](result/lab_03_model_loss.png)

### B. CIFAR-10 Results (`ex_1.ipynb`)
Due to the high complexity of CIFAR-10 (color variation, background clutter, and diverse objects), standard shallow MLPs struggle to achieve high performance. Our deep MLP utilized **Regularized Layers (L2)**, **Batch Normalization**, and **Dropout** alongside He Normal initialization to optimize training:
* **Epoch 1:** Train Accuracy = `32.74%` | Train Loss = `2.3004` | Val Accuracy = `28.81%` | Val Loss = `2.3614`
* **Epoch 4:** Train Accuracy = `43.01%` | Train Loss = `1.8302` | Val Accuracy = `38.45%` | Val Loss = `1.9183`

The introduction of **Batch Normalization** after each Dense layer re-scales the activations to have zero mean and unit variance, which complements He Normal initialization. This allows for stable gradient flow, higher learning rates, and accelerates model convergence significantly.
