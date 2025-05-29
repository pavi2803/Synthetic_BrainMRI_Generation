# 🧠 VQ-VAE Training on 3D Medical Images

This project implements a training pipeline for a **Vector Quantized Variational Autoencoder (VQ-VAE)** on 3D medical imaging data (e.g., brain MRIs in `.nii.gz` format). It uses **PyTorch**, **MONAI**, and **Weights & Biases** for logging.

---

## 📂 Project Structure

- `CustomNiftiDataset`: Custom dataset to load `.nii.gz` images from a CSV.
- `train_model`: Main function to train the VQ-VAE using AMP (mixed precision).
- `train_transform`: MONAI transforms to load, scale, crop, and preprocess 3D images.
- Logging via **WandB** and memory monitoring during training.

---

## 🧪 Requirements

- Python 3.8+
- PyTorch
- MONAI
- pandas
- wandb
- nibabel

Install dependencies:

```bash
bash
Copy code
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Prepare your dataset:
    - CSV file with a column (e.g., `image`) pointing to `.nii.gz` image paths.
2. Update the code with the path to your CSV and column name.
3. Start training:

```bash
bash
Copy code
python train.py
```

---

## 📈 Features

- Mixed precision training (`torch.cuda.amp`)
- L1 loss + vector quantization loss
- Support for GPU
- Clean memory usage logging
- Easily customizable MONAI transforms

---

## 📝 Sample Output

```
plaintext
Copy code
Epoch 1: recons_loss=0.023, quantization_loss=0.001
GPU Memory Allocated: 2000 MB
```
