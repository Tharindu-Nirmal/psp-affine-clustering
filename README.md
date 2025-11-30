# Affine Subspace Models and PSP Denoising

This repository provides code for the paper:

> **Affine Subspace Models and Clustering for Patch-Based Image Denoising**  
> Tharindu Wickremasinghe, Marco F. Duarte, 2025.

The main contribution is a **Patch Subspace Projection (PSP)** denoiser
that uses affine subspace clustering of image patches, comparing three
self-representation models:

- **BPDN** (no affineness)
- **NN Lasso** (approximate affineness)
- **NNC Lasso** (affineness via constrained non-negative lasso)

We also compare PSP against a Non-Local Means (NLM) baseline.

---

## 1. Installation

Tested with Python 3.10.

```bash
git clone https://github.com/<your-username>/psp-affine-denoising.git
cd psp-affine-denoising

# with venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
