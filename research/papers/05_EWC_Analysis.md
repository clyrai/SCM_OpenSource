# Paper Analysis: EWC (Elastic Weight Consolidation)

## Paper Details
- **Title**: Overcoming Catastrophic Forgetting in Neural Networks
- **arXiv**: 1612.00796
- **Authors**: James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshna Kumaran, Raia Hadsell
- **Institution**: Google DeepMind
- **Journal**: PNAS (Proceedings of the National Academy of Sciences), 2017
- **Citation**: PNAS 114(13) 3521-3526 (2017)

---

## 1. Summary

EWC is a foundational paper that addresses **catastrophic forgetting** in neural networks by protecting weights important for previously learned tasks. It uses Fisher information to identify important weights and adds a regularization term to prevent these weights from changing too much during new learning.

**Key Insight**: After learning a task, identify which weights are important (using Fisher information), then when learning a new task, penalize changes to those important weights.

---

## 2. The Problem: Catastrophic Forgetting

When neural networks learn sequentially:
- Learning task B overwrites weights learned for task A
- Performance on task A drops to near zero
- The network "forgets" old tasks completely

This is because standard gradient descent treats all weights equally — no protection for previously learned knowledge.

---

## 3. The Solution: Elastic Weight Consolidation

### 3.1 Core Idea

After learning task A, some weights become more important than others for that task. When learning task B:
- Allow weights important for A to change only slightly
- Allow other weights to change freely

### 3.2 The Algorithm

**Step 1: Learn Task A**
```
Train network on task A
θ_A* = optimal weights after training on A
```

**Step 2: Compute Importance**
```
Compute Fisher Information Matrix F
F[i] = E[(d log p(y|x,θ)/dθ_i)^2]  # Expected gradient squared
# High Fisher = important weight for task A
```

**Step 3: Learn Task B with Protection**
```
L_total = L_B(θ) + λ * Σ F_i * (θ_i - θ_A*_i)^2

Where:
  L_B(θ) = standard loss on task B
  λ = importance of task A vs task B
  Σ F_i * (θ_i - θ_A*_i)^2 = EWC penalty
```

The penalty says: "Don't change weights that were important for task A."

### 3.3 Fisher Information

Fisher information measures how much a weight affects the loss:
- **High Fisher**: Small change in weight → big change in loss → IMPORTANT
- **Low Fisher**: Weight can change without affecting loss → NOT IMPORTANT

```
F_i = E[(∂ log p(y|x,θ) / ∂θ_i)^2]
```

Approximated as: average squared gradient over data.

---

## 4. Why EWC Works

### 4.1 Intuition
- Weights with high Fisher are "critical" for old task
- Penalizing changes to these weights protects old knowledge
- Network can still learn new task with remaining flexibility

### 4.2 Mathematical Justification
EWC is derived from **Bayesian inference**:
- After learning task A, weights have posterior p(θ|D_A)
- Learning task B should update this posterior
- EWC approximates this with a Gaussian approximation
- Result: quadratic penalty on important weights

---

## 5. Key Results

### 5.1 Permuted MNIST
- Train on MNIST digits 0-4
- Then train on digits 5-9
- **Without EWC**: Forgets first task completely
- **With EWC**: Maintains performance on both

### 5.2 Reinforcement Learning
- Learn multiple RL tasks sequentially
- EWC preserves performance on old tasks
- Agent can learn new tasks without relearning old ones

### 5.3 Comparison to Other Methods
| Method | Performance |
|--------|-------------|
| No protection | ~10% (chance) on old task |
| EWC | ~70-90% on old task |
| Progressive networks | ~95% but grows with tasks |

---

## 6. Limitations

### 6.1 Computational Cost
- Fisher information matrix is O(n²) in parameters
- For large networks (millions of params), expensive
- Approximations needed for scale

### 6.2 Task Boundary Assumption
- Assumes clear boundaries between tasks
- Real learning is continuous, not task-based
- Doesn't handle task-agnostic learning well

### 6.3 Negative Interference
- Protecting old weights can slow new learning
- May accumulate with many tasks
- Trade-off between old and new performance

### 6.4 No Consolidation
- EWC protects weights, but doesn't reorganize them
- Memory stays in weights, not in structured memory
- No equivalent of memory transfer to cortex

### 6.5 No Sleep
- Purely online learning approach
- No offline consolidation phase
- No sleep-inspired mechanisms

---

## 7. What EWC Doesn't Have (For SleepAI)

### 7.1 No Meaning Encoding
- All knowledge is in weights
- No semantic representation
- Can't query "what do I know about X?"

### 7.2 No Active Forgetting
- EWC prevents change, but doesn't forget noise
- Memory grows unbounded
- No importance thresholding for deletion

### 7.3 No Sleep/Consolidation
- Single-phase learning
- No offline reorganization
- No Hebbian replay

### 7.4 No Hippocampal-Cortical Split
- All memory is weight-based
- No two-stage memory
- No fast/slow memory distinction

### 7.5 No Value/Emotional Tagging
- Importance = Fisher information
- No multi-dimensional importance
- All tasks treated equally

---

## 8. Relationship to Other Papers

### 8.1 EWC vs SleepGate
| Aspect | EWC | SleepGate |
|--------|-----|-----------|
| **Target** | Training (weights) | Inference (KV cache) |
| **Mechanism** | Regularization | Cache eviction |
| **When** | Between tasks | During inference |
| **Sleep-inspired** | No | Yes |
| **Forgetting** | Implicit (penalty) | Explicit (eviction) |

### 8.2 EWC vs MemGPT/Mem0
| Aspect | EWC | MemGPT/Mem0 |
|--------|-----|-------------|
| **Memory location** | Weights | External storage |
| **Retrieval** | Not applicable | Retrieval-based |
| **Scalability** | Limited by Fisher | Better |
| **Sleep** | No | No |

---

## 9. What SleepAI Can Borrow from EWC

1. **Importance weighting**: Not all memory is equally important
2. **Protection with flexibility**: Protect key memories, allow changes elsewhere
3. **Fisher-like signals**: Could use attention patterns as importance signal
4. **Quadratic penalty concept**: Memory has cost, trade-off between retention and learning

---

## 10. Quick Reference

**EWC = Weight Protection for Continual Learning**

| Brain Concept | EWC Implementation |
|--------------|---------------------|
| Synaptic protection | Fisher-weighted penalty |
| Important synapses | High Fisher weights |
| Memory cost | Quadratic regularization |
| Learning stability | Elastic (spring-like) weight change |

**Key Insight**: EWC's insight that some weights matter more than others maps to SleepAI's need for importance-based memory management. But EWC operates on weights, SleepAI needs to operate on semantic memory.

---

## 11. Mathematical Reference

**EWC Loss**:
```
L_total(θ) = L_B(θ) + λ * Σ_i F_A[i] * (θ_i - θ_A*[i])²
```

**Fisher Information Approximation**:
```
F[i] ≈ (1/N) Σ_k (∂ log p(y_k|x_k, θ) / ∂θ_i)²
```

**Recursive EWC** (for multiple tasks):
```
After task B, θ_B* becomes θ_A* for task C
Fisher accumulates: F_total = F_A + F_B
```

---

*Analysis Date: April 2026*
*Project: SleepAI*
