# Affine Subspace Models and PSP Denoising

This repository provides code for the paper:

> **Affine Subspace Models and Clustering for Patch-Based Image Denoising**  
> Tharindu Wickremasinghe, Marco F. Duarte, 2025.

[![arXiv](https://img.shields.io/badge/arXiv-2512.07259-b31b1b.svg)](https://arxiv.org/abs/2512.07259)

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
git clone https://github.com/Tharindu-Nirmal/psp-affine-clustering.git
cd psp-affine-clustering

# Create and activate the conda environment
conda env create -f environment.yml
conda activate psp_denoising

# Quick sanity check
python ssc_pipeline.py --dataset-dir Dataset/baseline_data --image-number 3 --solver elasticnet

```

### Folder structure
After cloning, the repo is organized as:
```bash
psp-affine-clustering/
├─ .gitignore
├─ psp_affine_clustering.py
├─ requirements.txt
├─ environment.yml
├─ NLM_Baseline.ipynb
├─ Dataset/
│ └─ baseline_data/
│   ├─ Image1.png
│   ├─ Image2.png
│   └─ ...
└─ results/ (created automatically; not tracked)
```

## 2. Run

### Quick sanity check
python psp_affine_clustering.py --dataset-dir Dataset/baseline_data --image-number 3 --solver elasticnet

### Choice of solvers
The three main solvers compared in the paper are:
1. bpdn (BPDN from the spgl1 library)   --> (BPDN in the paper)
2. elasticnet (Elastic net from the sklearn library) --> (NC LASSO in the paper)
3. lasso (LASSO from the spgl1 library) --> (NNC LASSO in the paper)

### Notes:
- If you plan to use --solver lasso or --solver bpdn, make sure spgl1 is installed in the environment.


## 3. Acknowledgements

We thank the IEEE Signal Processing Society (SPS) for creating the [ME-UYR mentoring program](https://signalprocessingsociety.org/tags/me-uyr-program) and for supporting this collaboration.

Our affine subspace clustering pipeline is adapted from Soltanolkotabi, Elhamifar, and Candès, “Robust Subspace Clustering,” *The Annals of Statistics*, 2014

## 4. Citation
If you feel this project helpful/insightful, please cite our paper:

```bibtex
@article{Tharindu2025_Affine,
  title   = {Affine Subspace Models and Clustering for Patch-Based Image Denoising},
  author  = {Wickremasinghe, Tharindu and Duarte, Marco F.},
  journal = {Proceedings of the Asilomar Conference on Signals, Systems and Computers},
  year    = {2025}
}
```

## 5. Contact
If you have any comments or questions, feel free to reach out on my email (lwickrem@purdue.edu).