# Autoencoders for Anomaly Detection

This repository contains two projects that apply **autoencoders** for **unsupervised anomaly detection** in different data domains: images and time series.  
The goal is to identify abnormal patterns based on reconstruction errors using deep learning models implemented in **PyTorch**.

---

## Overview

| Notebook | Domain | Description |
|-----------|---------|-------------|
| `autoencoder_mnist.ipynb` | Image | Autoencoder trained on MNIST to demonstrate reconstruction-based anomaly detection. |
| `autoencoder_timeseries.ipynb` | Time series | Autoencoder applied to synthetic industrial-style time-series data using quantile and precision–recall thresholding. |

Both notebooks are designed to illustrate the use of autoencoders for detecting deviations without labeled data.

---

## Methods

- **Framework:** PyTorch  
- **Techniques:** Dimensionality reduction, reconstruction error analysis, quantile/PR-based thresholding  
- **Metrics:** MSE, precision, recall, F1-score  
- **Libraries:** NumPy, pandas, matplotlib, scikit-learn

---

## Results

- The MNIST model accurately reconstructs normal digits and identifies unseen or distorted samples as anomalies.  
- The time-series model achieves stable detection of synthetic anomalies with minimal false positives by adjusting detection thresholds based on PR curves.

---

## Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn
