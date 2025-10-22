# 🧭 Optimization Algorithms in Machine Learning

## 1. Goal of Optimization

Optimization algorithms **minimize the loss function** by updating model parameters (weights).  
They determine how fast and in what direction the weights move to reach the minimum of the loss surface.

---

## 2. Classic Gradient Descent (GD)

**Update rule:**

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

Where:  
- $\eta$: learning rate  
- $\nabla L(w_t)$: gradient of loss w.r.t. weights  

**Pros:**
- Simple and deterministic  
- Theoretical convergence guarantees for convex problems  

**Cons:**
- Computationally expensive for large datasets  
- Can get stuck in local minima  
- Sensitive to learning rate  

**Use Case:**  
Small datasets, convex loss functions (e.g., Linear Regression).

---

## 3. Stochastic Gradient Descent (SGD)

**Idea:** Updates weights after **every training example** instead of the whole dataset.

**Update rule:**

$$
w_{t+1} = w_t - \eta \nabla L_i(w_t)
$$

where $L_i$ is the loss for the $i$-th training example.

**Pros:**
- Fast and efficient for large datasets  
- Can escape shallow local minima  

**Cons:**
- High variance → noisy convergence  
- Requires careful learning rate tuning  

**Use Case:**  
Online learning, deep neural networks, large-scale datasets.

---

## 4. Mini-Batch Gradient Descent

**Idea:**  
Uses a **small batch of samples** (size $m$) for each update.

**Update rule:**

$$
w_{t+1} = w_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla L_i(w_t)
$$

**Pros:**
- Efficient GPU parallelization  
- Balances stability and speed  
- Smoother convergence than SGD  

**Cons:**
- Still sensitive to learning rate  
- Requires batch size tuning  

**Use Case:**  
Deep learning models; the standard approach in modern training pipelines.

---

## 5. Momentum-Based Optimization

**Concept:**  
Incorporates past gradients to smooth updates and accelerate convergence.

**Update rule:**

$$
v_{t+1} = \beta v_t + (1 - \beta) \nabla L(w_t)
$$

$$
w_{t+1} = w_t - \eta v_{t+1}
$$

Where:  
- $v_t$: velocity (running average of gradients)  
- $\beta$: momentum factor (≈ 0.9)  
- $\eta$: learning rate  

**Pros:**
- Faster convergence  
- Reduces oscillations  
- Helps escape local minima  

**Cons:**
- May overshoot if $\beta$ too high  

**Use Case:**  
Convex/non-convex problems; helps accelerate SGD.

---

## 6. Nesterov Accelerated Gradient (NAG)

**Idea:**  
Looks ahead — computes gradient at the *future* position of the parameters.

**Update rule:**

$$
v_{t+1} = \beta v_t + \nabla L(w_t - \eta \beta v_t)
$$

$$
w_{t+1} = w_t - \eta v_{t+1}
$$

**Pros:**
- Anticipates future gradient direction  
- Faster and more stable convergence  

**Cons:**
- Slightly more computation  

**Use Case:**  
Training deep networks where oscillations are high.

---

## 7. Adagrad (Adaptive Gradient Algorithm)

**Idea:**  
Each parameter has its own adaptive learning rate that decreases over time.

**Update rule:**

$$
G_t = G_{t-1} + (\nabla_\theta J(\theta))^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)
$$

**Pros:**
- Works well with sparse data  
- Requires little tuning  

**Cons:**
- Learning rate decays too fast → may stop learning  

**Use Case:**  
Sparse data problems (e.g., NLP, recommender systems).

---

## 8. RMSProp (Root Mean Square Propagation)

**Idea:**  
Fixes Adagrad’s vanishing learning rate by using an exponentially decaying average.

**Update rule:**

$$
G_t = \gamma G_{t-1} + (1 - \gamma)(\nabla_\theta J(\theta))^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)
$$

**Pros:**
- Maintains stable learning rate  
- Works well with non-stationary objectives  

**Cons:**
- Sensitive to decay rate $\gamma$  

**Use Case:**  
RNNs, non-stationary or online learning tasks.

---

## 9. AdaDelta

**Idea:**  
Improves Adagrad by limiting accumulated gradients and using moving averages.

**Update rule:**

$$
\Delta \theta_{t+1} = - \frac{\sqrt{E[\Delta \theta^2]_t + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

**Pros:**
- Stable learning rate  
- No need to set a default learning rate  

**Cons:**
- Slightly more complex implementation  

**Use Case:**  
When learning rate scheduling is tricky.

---

## 10. AdaMomentum

**Idea:**  
Combines adaptive learning rates (Adagrad) with momentum memory.

**Pros:**
- Adaptive and stable updates  
- Balances speed and precision  

**Cons:**
- Computationally heavier  

**Use Case:**  
Fine-tuning large models where both stability and adaptivity matter.

---

## 11. Adam (Adaptive Moment Estimation)

**Idea:**  
Combines Momentum + RMSProp.  
Maintains moving averages of both gradients and squared gradients.

**Update equations:**

First moment (mean):
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla L(w_t)
$$

Second moment (variance):
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla L(w_t))^2
$$

Bias correction:
$$
\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1 - \beta_2^t}
$$

Weight update:
$$
w_{t+1} = w_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
$$

**Typical parameters:**
- $\beta_1 = 0.9$  
- $\beta_2 = 0.999$  
- $\epsilon = 10^{-8}$  
- $\eta = 0.001$

**Pros:**
- Fast and robust  
- Works well with noisy data  
- Minimal tuning required  

**Cons:**
- May overfit  
- Sometimes generalizes worse than SGD  

**Use Case:**  
Default choice for most deep learning models.

---

## 12. Comparison Summary

| Optimizer | Momentum | Adaptive LR | Pros | Cons | Use Case |
|------------|-----------|--------------|------|------|-----------|
| **GD** | ❌ | ❌ | Deterministic, simple | Slow on large data | Small convex problems |
| **SGD** | ❌ | ❌ | Fast, stochastic | Noisy updates | Online learning |
| **Mini-Batch GD** | ❌ | ❌ | Stable + efficient | Needs batch tuning | Standard DL setup |
| **Momentum** | ✅ | ❌ | Smooths convergence | May overshoot | Convex loss |
| **NAG** | ✅ | ❌ | Anticipates gradient | Costlier | Deep networks |
| **Adagrad** | ❌ | ✅ | Sparse-friendly | LR decays too fast | NLP tasks |
| **RMSProp** | ✅(light) | ✅ | Works with RNNs | Sensitive to γ | Non-stationary tasks |
| **AdaDelta** | ✅ | ✅ | Stable learning rate | Complex | Adaptive scheduling |
| **Adam** | ✅ | ✅ | Fast, reliable | May overfit | General use |
| **AdaMomentum** | ✅ | ✅ | Adaptive + stable | Heavy compute | Fine-tuning |

---

## 13. Practical Tips

- Use **Adam** by default.  
- Try **Momentum** or **NAG** for convex/simple problems.  
- Use **RMSProp** for RNNs or time-series tasks.  
- Use **Adagrad** for sparse NLP problems.  
- Always **tune the learning rate ($\eta$)** first.

---
