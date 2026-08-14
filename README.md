# Spectral Certification for Class-Level Machine Unlearning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of the **Residual Irreversibility Index (RII)** — a spectral certificate for quantifying output-space irreversibility in class-level machine unlearning.

**Paper**: *"Spectral Certification for Class-Level Machine Unlearning"*  
**Submitted to**: Applied Intelligence (Springer)

**Code**: [GitHub](https://github.com/EurusDemerzel/rii_unlearning) | [Gitee](https://gitee.com/peregrine_eurus/rii_unlearning)

---

## Overview

The central object is a **$2 \times C$ empirical output channel matrix** whose rows are the averaged softmax predictions on the forget set $D_f$ and retain set $D_r$. We define the **Residual Irreversibility Index (RII)**:

$$\rho = 1 - \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} \in [0, 0.5]$$

from the singular values of this matrix. $\rho = 0$ indicates perfect unlearning (rank-one channel), while $\rho > 0$ reveals residual information leakage.

For class-level forgetting, we introduce the **Multi-Held-Out Projection Residual (MHPR)** $\rho_H$, which projects the forgotten class mean onto the subspace spanned by multiple unseen-class means.

### Key Results

| Dataset | Classes | Accuracy | Random Forget $\rho$ | Class-Level $\rho$ | MHPR $\rho_H$ |
|---------|:-------:|:--------:|:--------------------:|:------------------:|:-------------:|
| MNIST | 10 | 97.5% | $9.8 \times 10^{-4}$ | 0.136 | **0.046** (K=3) |
| Fashion-MNIST | 10 | 88.7% | $2.83 \times 10^{-4}$ | 0.102 | **0.021** (K=9, T=50) |
| CIFAR-10 | 10 | 75.0% | $1.2 \times 10^{-3}$ | 0.197 | **0.025** (K=9, T=5) |
| CIFAR-100 | 100 | 40.6% | $7.1 \times 10^{-4}$ | 0.010 | 0.133 (K=10) |
| Tiny ImageNet | 200 | 28.4% | $1.35 \times 10^{-3}$ | — | — |

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with MPS or CUDA recommended)
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/EurusDemerzel/rii_unlearning.git
cd rii_unlearning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from models import get_model
from metrics import compute_rii_from_probs

# Load a pre-trained model
model = get_model('mnist', torch.device('cpu'), model_name='mlp')

# ... obtain softmax predictions for forget and retain sets ...
# pf: (N_f, C) softmax probabilities on forget set
# pr: (N_r, C) softmax probabilities on retain set

rho, mi_ub = compute_rii_from_probs(pf, pr)
print(f"RII: {rho:.6f}, MI upper bound: {mi_ub:.6f} nats")
```

## Repository Structure

```
one_rank/
├── metrics.py                 # RII and MHPR computation
├── models.py                  # Model definitions (MLP, CNN, ResNet)
├── pipeline.py                # Full experiment pipeline
├── unlearn.py                 # Unlearning algorithms
├── sisa_unlearning.py         # SISA implementation
├── targeted_forget.py         # Targeted forgetting
├── machine_unlearning.py      # Additional utilities
├── visualize.py               # Plotting utilities
├── config.yaml                # Experiment configuration
├── requirements.txt           # Dependencies
├── setup.sh                   # Environment setup script
│
├── run_fashion_mnist.py       # Fashion-MNIST experiments
├── run_multi_heldout_projection.py  # MHPR experiments
├── run_cifar100_long.py       # CIFAR-100 long training
├── run_cifar10_multi_class_forget.py # CIFAR-10 class forgetting
├── run_abcde.py               # Supplemental experiments
│
├── main1.tex                  # Paper (LaTeX source)
├── IEEEtran.cls               # IEEEtran document class
├── IEEEtran.bst               # IEEEtran bibliography style
│
├── LICENSE                    # MIT License
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Reproducing the Paper

### Visual benchmarks (CIFAR-10, Fashion-MNIST, CIFAR-100)

```bash
source venv/bin/activate

# CIFAR-10 class-level benchmark (7 methods, Table tab:benchmark)
python benchmark_v2.py --seed 0

# Fashion-MNIST cross-dataset benchmark (Table tab:cross)
python benchmark_v2.py --fashion_mnist --seed 0

# CIFAR-100 class-level benchmark (Table tab:cifar100)
python benchmark_cifar100.py --seed 0

# MNIST MHPR evaluation and K-ablation (Table tab:mhpr, Fig. k_ablation)
python run_multi_heldout_projection.py
```

### Language-model experiments

```bash
# TOFU author-level on LLaMA-2-7B (Table tab:llm) — requires mlx + mlx-lm
python benchmark_llm_tofu_mlx.py --method fine_tune
python benchmark_llm_tofu_mlx.py --method neggrad
python benchmark_llm_tofu_mlx.py --method retrain

# NegGrad strength sweep (Sec. 5.3)
python run_neggrad_scan.py

# AG News + DistilBERT pilot (cross-modal evidence)
python benchmark_nlp.py
```

### Compiling the paper (Springer svjour3 template)

```bash
cd ai_submission
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Environment

- Python 3.10+, PyTorch 2.0+ (MPS or CUDA)
- `requirements.txt` — vision experiments
- `mlx` + `mlx-lm` — LLM experiments (Apple Silicon)
- `environment.yml` (conda) mirrors the above
pdflatex main1.tex
```

## Datasets

The following datasets are used (automatically downloaded by `torchvision` on first use):

| Dataset | Source | Classes | Train/Test |
|---------|--------|:-------:|:----------:|
| MNIST | `torchvision.datasets.MNIST` | 10 | 60K/10K |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | 10 | 60K/10K |
| CIFAR-10 | `torchvision.datasets.CIFAR10` | 10 | 50K/10K |
| CIFAR-100 | `torchvision.datasets.CIFAR100` | 100 | 50K/10K |
| Tiny ImageNet | Manual download | 200 | 100K/10K |

For Tiny ImageNet, download from [tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com/) and place in `data/tiny-imagenet-200/`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhang2026spectral,
  title={Spectral Certification for Class-Level Machine Unlearning},
  author={Zhang, Yan and Lu, Chenggang},
  journal={Applied Intelligence},
  year={2026},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Yan Zhang — ZJUT, College of Mathematical Sciences  
For questions or collaborations, please open an issue on GitHub.
