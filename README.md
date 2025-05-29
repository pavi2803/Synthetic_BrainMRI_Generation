## VQ-VAE for Synthetic Brain MRI Generation
This project focuses on generating high-quality synthetic brain MRI scans using the Vector Quantized Variational Autoencoder (VQ-VAE) architecture to address data scarcity in neuroimaging research, particularly for dementia diagnosis. 


This repository contains code for training and evaluating a Vector Quantized Variational Autoencoder (VQ-VAE) model to generate synthetic brain MRI images. The project uses the MONAI framework for medical imaging deep learning and incorporates preprocessing, training, and evaluation pipelines.

Table of Contents
Overview

Installation

Dataset

Usage

Model Architecture

Evaluation Metrics

Contributing

License

Overview
This project implements a VQ-VAE model for generating synthetic brain MRI images. It utilizes advanced preprocessing techniques and MONAI's utilities to handle NIfTI medical image formats. The training and evaluation pipeline includes:

Data loading and preprocessing from Excel metadata and NIfTI files

Custom dataset classes for flexible data handling

Use of mixed precision training and GPU acceleration

Evaluation with metrics like FID, MMD, SSIM, and multi-scale SSIM

Experiment tracking with Weights & Biases (WandB)

Installation
The repository requires Python 3.7+ and the following key dependencies:

bash
Copy
Edit
pip install torch torchvision torchaudio
pip install monai-weekly[tqdm,nibabel]
pip install matplotlib pandas scikit-learn wandb openpyxl nibabel
Alternatively, use the included package install script in the notebook:

python
Copy
Edit
def install_packages():
    try:
        import monai
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "monai-weekly[tqdm, nibabel]"])

    try:
        import matplotlib
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
Dataset
The data is loaded from an Excel spreadsheet containing metadata and paths to NIfTI MRI images. The dataset is split into training, validation, and test sets, ensuring no subject overlap across these splits.

Dataset source path:

/ifs/loni/faculty/thompson/four_d/ADNI/Spreadsheets_ADNI/ADNI_all_T1_DLpaths_DWIpaths_demographics_20240418_shared.xlsx

Usage
Prepare Data:

Filter and preprocess dataset based on scan counts and valid image paths.

Validate existence of NIfTI files.

Create Custom Dataset:

Use CustomNiftiDataset class to load and transform data for PyTorch.

Initialize GPU:

python
Copy
Edit
device = initialize_gpu(0)  # Choose GPU device
Train and Evaluate VQ-VAE:

Use MONAI's transforms for data augmentation and normalization.

Train using your preferred training loop.

Track metrics with WandB.

Model Architecture
The VQ-VAE architecture is implemented in the generative.networks.nets.VQVAE module. It is designed to compress and reconstruct brain MRI images using vector quantization techniques, enabling efficient latent space representation for generative modeling.

Evaluation Metrics
Model performance is evaluated using:

Fréchet Inception Distance (FID)

Maximum Mean Discrepancy (MMD)

Structural Similarity Index Metric (SSIM)

Multi-Scale SSIM (MS-SSIM)

These metrics quantitatively assess the quality and diversity of synthetic images compared to real data.

Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

License
MIT License

