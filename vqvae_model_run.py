import sys
print(sys.executable)

def install_packages():
    try:
        import monai
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "monai-weekly[tqdm, nibabel]"])

    try:
        import matplotlib
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

install_packages()

import sys
import os
print("Before chdir:", os.getcwd())

try:
    os.chdir('/ifshome/psenthilkumar/SYN/synth_2')
    print("After chdir:", os.getcwd())
except Exception as e:
    print(f"chdir() failed: {e}")

from generative.networks.nets import VQVAE
print(VQVAE)
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from generative.networks.nets import VQVAE
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from torch.cuda.amp import GradScaler, autocast
print_config()

from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    Lambdad,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CenterSpatialCropd,
    Resized,
)


import wandb  # Import WandB
import pandas as pd
import numpy as np
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric


def initialize_gpu(num):
    device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return device

def load_and_preprocess_data(path):
    
    import pandas as pd
    import numpy as np
    path="/ifs/loni/faculty/thompson/four_d/ADNI/Spreadsheets_ADNI/ADNI_all_T1_DLpaths_DWIpaths_demographics_20240418_shared.xlsx" 
    data = pd.read_excel(path, engine='openpyxl')
    
    
    data_filtered = data[data['ORIGPROT'] != 'ADNI3']

    data_1 = data_filtered.groupby('SubjID').agg(
        Scan_Count=('SubjID', 'size'),  # Count the number of scans
        Paths=('NONACCEL_DL_9DOF_2MM_T1', lambda x: list(x))  # Aggregate paths into a list
    ).reset_index()
    
    
    filtered_data = data_1[(data_1['Scan_Count'] > 0) & (data_1['Paths'].apply(lambda x: any(pd.notna(x))))]

    # Print the filtered DataFrame
    print(filtered_data)
    
    
    subjects_with_one_scan = data_1.loc[data_1['Scan_Count'] == 1, ['SubjID', 'Paths']]
    print(subjects_with_one_scan)

    filtered_subjects_one_scan = subjects_with_one_scan[
        subjects_with_one_scan['Paths'].apply(lambda x: isinstance(x, list) and not all(isinstance(i, float) and np.isnan(i) for i in x))
    ]['SubjID'].tolist()

    print(len(filtered_subjects_one_scan)) 


    data_1['Image'] = data_1['Paths'].apply(lambda x: x[0] if len(x) > 0 else None)

    print(len(data_1))

    data_2 = data_1.dropna(subset=['Image'])

    # Print the resulting DataFrame
    
    unique_subjects_distribution = data_2.groupby('Scan_Count')['SubjID'].nunique()

    # Display results
    print("Total subjects per Scan_Count:")
    unique_subjects_distribution
    
    data_2_view = data_2[data_2['Scan_Count']==1]

    data_2_view.head(5)

    for i in data_2_view['Paths']:
        print(i)
        
        
    test_dataset = data_2[data_2['SubjID'].isin(filtered_subjects_one_scan)]
    
    test_subjects_scan_1 = data_2[data_2['Scan_Count'] == 1]

    # Get remaining subjects with Scan_Count > 1 (871 scans required to make total 1000)
    remaining_data = data_2[data_2['Scan_Count'] > 1]
    additional_test_data = remaining_data.sample(n=300 - len(test_subjects_scan_1), random_state=42)

    # Combine both to create the final test dataset
    test_data = pd.concat([test_subjects_scan_1, additional_test_data])

    # Step 2: Exclude test subjects to get remaining data for train/val split
    remaining_data_for_train_val = data_2[~data_2['SubjID'].isin(test_data['SubjID'])]

    # Step 3: Select Validation Set (300 scans), ensuring no overlap with test subjects
    val_data = remaining_data_for_train_val.sample(n=100, random_state=42)

    # Step 4: Get unique subjects from validation set and exclude them from training
    val_subjects = val_data['SubjID'].unique().tolist()
    train_data = remaining_data_for_train_val[~remaining_data_for_train_val['SubjID'].isin(val_subjects)]

    # Step 5: Verify sizes
    print(f"Test Samples: {len(test_data)} (Subjects: {test_data['SubjID'].nunique()})")
    print(f"Validation Samples: {len(val_data)} (Subjects: {len(val_subjects)})")
    print(f"Training Samples: {len(train_data)} (Subjects: {train_data['SubjID'].nunique()})")
    
    
    test_scans = test_data['Scan_Count'].sum()
    val_scans = val_data['Scan_Count'].sum()
    train_scans = train_data['Scan_Count'].sum()

    # Print final counts
    print(f"Total Test Scans: {test_scans}")
    print(f"Total Validation Scans: {val_scans}")
    print(f"Total Training Scans: {train_scans}")
    
    
    all_unique_members = [i for i in data_2['SubjID']]
    all_unique_members = list(data_2['SubjID'].unique())
    
    
    common_subjects = set(train_data['SubjID']).intersection(set(val_data['SubjID']))


    if common_subjects:
        print(f"Found {len(common_subjects)} common subject IDs between train and validation datasets.")
        print("Common Subject IDs:", common_subjects)
    else:
        print("No common subject IDs found between train and validation datasets.")

        
    train_dataset = train_data[['Image']]
    val_dataset = val_data[['Image']]
    test_dataset = test_data[['Image']]
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    
    
    train_dataset = train_dataset.dropna(subset=['Image'])
    val_dataset = val_dataset.dropna(subset=['Image'])
    
    set_determinism(42)
    
    

    # Function to check if the file exists
    def check_file_exists(file_path):
        return os.path.isfile(file_path)


    valid_data = []

    # Check if each file exists in the dataframe and create a list of valid files
    for idx, row in train_dataset.iterrows():
        image_path = row['Image']
        if isinstance(image_path, np.ndarray):
            image_path = image_path.item()  # Convert numpy array to string if necessary

        if check_file_exists(image_path):
            valid_data.append(row)
        else:
            print(f"Warning: File not found in Train: {image_path}")

    # If no valid files are found, raise an exception or handle accordingly
    if not valid_data:
        raise ValueError("No valid NIfTI files found in the dataset.")

    # Convert the valid data back to a DataFrame
    train_dataset = pd.DataFrame(valid_data)

    print("train data created")


    valid_data = []

    # Check if each file exists in the dataframe and create a list of valid files
    for idx, row in test_dataset.iterrows():
        image_path = row['Image']
        if isinstance(image_path, np.ndarray):
            image_path = image_path.item()  # Convert numpy array to string if necessary

        if check_file_exists(image_path):
            valid_data.append(row)
        else:
            print(f"Warning: File not found in Test: {image_path}")

    if not valid_data:
        raise ValueError("No valid NIfTI files found in the dataset.")


    test_dataset = pd.DataFrame(valid_data)
    print("Test data created")

    valid_data = []

    for idx, row in val_dataset.iterrows():
        image_path = row['Image']
        if isinstance(image_path, np.ndarray):
            image_path = image_path.item() 

        if check_file_exists(image_path):
            valid_data.append(row)
        else:
            print(f"Warning: File not found in Validation: {image_path}")


    if not valid_data:
        raise ValueError("No valid NIfTI files found in the dataset.")


    val_dataset = pd.DataFrame(valid_data)

    print("Validation Data created")


    class CustomNiftiDataset(Dataset):
        def __init__(self, dataframe, image_column, transform=None):
            self.dataframe = dataframe
            self.image_column = image_column
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            # Load the image path from the DataFrame
            image_path = self.dataframe.iloc[idx][self.image_column]

            # Ensure the image path is a string, not a numpy array
            if isinstance(image_path, np.ndarray):
                image_path = image_path.item()  # Convert numpy array to string if necessary

            # Prepare the sample as a dictionary
            sample = {"image": image_path}

            # Apply transformations (LoadImaged expects the file path)
            if self.transform:
                sample = self.transform(sample)

            # Ensure the image has a batch dimension
            if isinstance(sample["image"], torch.Tensor):
                sample["image"] = sample["image"].unsqueeze(0)  # Add batch dimension

            return sample

## LARGER SPATIAL DIM

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.Lambdad(keys="image", func=lambda x: x),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            transforms.ScaleIntensityd(keys=["image"]),
            transforms.RandSpatialCropd(keys=["image"], roi_size=[80, 92, 80]),  # Ensure the crop size matches the 3D image size

            # transforms.Resized(keys=["image"], spatial_size=(80, 84, 80)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.Lambdad(keys="image", func=lambda x: x),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            transforms.ScaleIntensityd(keys=["image"]),
            transforms.RandSpatialCropd(keys=["image"], roi_size=[80, 92, 80]),  # Same crop size for validation

            # transforms.Resized(keys=["image"], spatial_size=(80, 84, 80)),
        ]
    )

## SMALLER SPATIAL DIM

#     train_transform = transforms.Compose(
#       [
#         transforms.LoadImaged(keys=["image"]),
#         transforms.Lambdad(keys="image", func=lambda x: x),
#         transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
#         transforms.ScaleIntensityd(keys=["image"]),
#         transforms.CenterSpatialCropd(keys=["image"], roi_size=[176, 224, 155]),
#         transforms.Resized(keys=["image"], spatial_size=(48,64,48)),
#     ]
#     )

#     val_transform = transforms.Compose(
#       [
#         transforms.LoadImaged(keys=["image"]),
#         transforms.Lambdad(keys="image", func=lambda x: x),
#         transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
#         transforms.ScaleIntensityd(keys=["image"]),
#         transforms.CenterSpatialCropd(keys=["image"], roi_size=[176, 224, 155]),
#         transforms.Resized(keys=["image"], spatial_size=(48,64,48)),
#     ]
#     )



    # Create datasets for train, val, and test
    train_ds = CustomNiftiDataset(dataframe=train_dataset, image_column='Image', transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8)

    sample = train_ds[0] # Fetch first samplec
    print(sample['image'].shape)  # Check if it’s a string or a tensor



    val_ds = CustomNiftiDataset(dataframe=val_dataset, image_column='Image', transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    print("val  created")

    test_ds = CustomNiftiDataset(dataframe=test_dataset, image_column='Image', transform=val_transform)  # Test uses val_transform
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory= True)

    print("train_ds  created")
    
    
    return train_loader, test_loader, val_loader

    

# Print the resulting DataFrame
#     data = pd.read_excel(path, engine='openpyxl')
#     data_1 = data.groupby('SubjID').agg(
#         Scan_Count=('SubjID', 'size'),
#         Paths=('NONACCEL_DL_9DOF_2MM_T1', lambda x: list(x))
#     ).reset_index()
    
#     filtered_data = data_1[(data_1['Scan_Count'] > 0) & (data_1['Paths'].apply(lambda x: any(pd.notna(x))))]
#     subjects_with_one_scan = data_1.loc[data_1['Scan_Count'] == 1, ['SubjID', 'Paths']]
    
#     filtered_subjects_one_scan = subjects_with_one_scan[
#         subjects_with_one_scan['Paths'].apply(lambda x: isinstance(x, list) and not all(isinstance(i, float) and np.isnan(i) for i in x))
#     ]['SubjID'].tolist()
    
#     data_1['Image'] = data_1['Paths'].apply(lambda x: x[0] if len(x) > 0 else None)
#     data_2 = data_1.dropna(subset=['Image'])
    
#     test_subjects_scan_1 = data_2[data_2['Scan_Count'] == 1]
#     remaining_data = data_2[data_2['Scan_Count'] > 1]
#     additional_test_data = remaining_data.sample(n=300 - len(test_subjects_scan_1), random_state=42)
#     test_data = pd.concat([test_subjects_scan_1, additional_test_data])
    
#     remaining_data_for_train_val = data_2[~data_2['SubjID'].isin(test_data['SubjID'])]
#     val_data = remaining_data_for_train_val.sample(n=100, random_state=42)
#     val_subjects = val_data['SubjID'].unique().tolist()
#     train_data = remaining_data_for_train_val[~remaining_data_for_train_val['SubjID'].isin(val_subjects)]
    
#     test_scans = test_data['Scan_Count'].sum()
#     val_scans = val_data['Scan_Count'].sum()
#     train_scans = train_data['Scan_Count'].sum()

#     # Print final counts
#     print(f"Total Test Scans: {test_scans}")
#     print(f"Total Validation Scans: {val_scans}")
#     print(f"Total Training Scans: {train_scans}")
#     return train_data[['Image']], val_data[['Image']], test_data[['Image']]

def train_model(train_loader, val_loader):
    import torch
    from tqdm import tqdm
    import time
    import os
    import torchvision.utils as vutils
    from torch.cuda.amp import GradScaler, autocast
    import wandb  # Import WandB
    from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric

    # Initialize WandB
    wandb.init(project='vqvae_train_experiments_mar20', name='run_0.1', reinit=True)

    mmd = MMDMetric()

    # Hyperparameters
    n_epochs = 200
    val_interval = 30
    n_example_images = 3
    epoch_recon_loss_list = []
    epoch_quant_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []

    # Log hyperparameters to WandB
    wandb.config.update({
        "n_epochs": n_epochs,
        "val_interval": val_interval,
        # Include other hyperparameters like learning rate if needed
    })

    # Timer for the total training time
    total_start = time.time()

    # Set the device to GPU (CUDA) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If no GPU is available, raise an error
    if device.type != 'cuda':
        raise RuntimeError("CUDA device is not available. Please check your GPU setup.")

    # Create a GradScaler for mixed precision training
    scaler = GradScaler()

    # Move the model to GPU
    model.to(device)

    # Start training
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            if batch is None or "image" not in batch or batch["image"] is None:
                print(f"Warning: Skipping step {step} due to missing data.")
                continue

            # Move images to GPU and remove unnecessary dimension
            images = batch["image"].to(device)
            images = images.squeeze(2)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training with autocast
            with autocast(enabled=True):  # Use autocast only during the forward pass
                reconstruction, quantization_loss = model(images=images)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                loss = recons_loss + quantization_loss.float()

            # Check for NaN/Inf loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping step {step} due to NaN/Inf loss.")
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()

            # Track losses
            epoch_loss += recons_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": recons_loss / (step + 1),
                    "quantization_loss": quantization_loss.item() / (step + 1),
                }
            )

            # Free up GPU memory
            torch.cuda.empty_cache()

            # Track memory (optional)
            mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            print(f"Step {step} - GPU Memory Allocated: {mem_alloc:.2f} MB, Reserved: {mem_reserved:.2f} MB")

            # Log metrics to WandB
            wandb.log({
                'reconstruction_loss': recons_loss.item(),
                'quantization_loss': quantization_loss.item(),
                'epoch_loss': epoch_loss / (step + 1),
            })

        # Store losses
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

        # GPU Memory Tracking
        mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"Epoch {epoch} - GPU Memory Allocated: {mem_alloc:.2f} MB, Reserved: {mem_reserved:.2f} MB")

        # Log epoch memory stats to WandB
        wandb.log({
            'mem_alloc': mem_alloc,
            'mem_reserved': mem_reserved,
        })
        
        save_dir = "reconstructed_images"
        os.makedirs(save_dir, exist_ok=True)

        # Validation every 'val_interval' epochs
        # Validation every 'val_interval' epochs
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            mmd_scores = []  # List to store MMD scores for the epoch
            ms_ssim_recon_scores = []  # List to store MS-SSIM scores for the epoch

            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)
                    images = images.squeeze(2)

                    # Forward pass with autocast for mixed precision
                    with autocast(enabled=True):  # Use autocast during validation as well
                        reconstruction, quantization_loss = model(images=images)
                        recons_loss = l1_loss(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()

                    # Compute MMD scores
                    for i in range(images.shape[0]):  # Iterate over batch
                        mmd_scores.append(mmd(images[i].float(), reconstruction[i].float()))

                    # Compute MS-SSIM scores
                    for i in range(images.shape[0]):  # Iterate over batch
                        ms_ssim_recon_scores.append(ms_ssim(images[i].float(), reconstruction[i].float()))
                        
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:n_example_images, 0].detach())
                        
                        
                    print("intermediatery images ",len(intermediary_images))
                    
                    break

                # Average validation loss
                val_loss /= val_step
                val_recon_epoch_loss_list.append(val_loss)

                # Compute and log MMD scores
                mmd_scores = torch.stack(mmd_scores)
                mmd_mean = mmd_scores.mean().item()
                mmd_std = mmd_scores.std().item()
                print(f"Epoch {epoch} - MMD score: {mmd_mean:.4f} ± {mmd_std:.4f}")

                # Compute and log MS-SSIM scores
                ms_ssim_scores = torch.stack(ms_ssim_recon_scores)
                ms_ssim_mean = ms_ssim_scores.mean().item()
                ms_ssim_std = ms_ssim_scores.std().item()
                print(f"Epoch {epoch} - MSSIM score: {ms_ssim_mean:.4f} ± {ms_ssim_std:.4f}")

                # Log validation metrics to WandB
                wandb.log({
                    'val_loss': val_loss,
                    'mmd_score': mmd_mean,
                    'mmd_score_std': mmd_std,
                    'ms_ssim_score': ms_ssim_mean,
                    'ms_ssim_score_std': ms_ssim_std,
                })

                output_dir_images = 'intermediary_images'
                os.makedirs(output_dir_images, exist_ok=True)

                val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))

                # Create subplots: One row per validation sample
                fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True, figsize=(10, len(val_samples) * 3))

                # Ensure ax is an iterable (handles single-row case)
                if len(val_samples) == 1:
                    ax = [ax]

#                 for image_n in range(len(val_samples)):
#                     reconstructions = intermediary_images[0]  # Extract reconstruction at this epoch
#                     print(reconstructions.shape)  # Debugging output to confirm shape

#                     # Concatenate slices in 3 orthogonal views
#                     reconstructions = np.concatenate(
#                         [
#                             reconstructions[0, :, :, reconstructions.shape[-1] // 2],  # Mid-depth slice
#                             reconstructions[0, :, reconstructions.shape[-2] // 2, :].T,  # Mid-height slice
#                             reconstructions[0, reconstructions.shape[-3] // 2, :, :].T,  # Mid-width slice
#                         ],
#                         axis=1,
#                     )

#                     # Plot each reconstruction
#                     ax[image_n].imshow(reconstructions, cmap="gray")
#                     ax[image_n].set_xticks([])
#                     ax[image_n].set_yticks([])
#                     ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")

#                 # Save the figure to the specified folder
#                 plt.savefig(os.path.join(output_dir_images, 'intermediary_images.png'))
#                 plt.close()  # Close the plot to avoid displaying it
                
                
                
                for image_n, epoch in enumerate(val_samples):
                    reconstructions = intermediary_images[0]  # Extract reconstruction at this epoch
                    print(reconstructions.shape)  # Debugging output to confirm shape

                    # Concatenate slices in 3 orthogonal views
                    reconstructions = np.concatenate(
                        [
                            reconstructions[0, :, :, reconstructions.shape[-1] // 2],  # Mid-depth slice
                            reconstructions[0, :, reconstructions.shape[-2] // 2, :].T,  # Mid-height slice
                            reconstructions[0, reconstructions.shape[-3] // 2, :, :].T,  # Mid-width slice
                        ],
                        axis=1,
                    )

                    # Plot each reconstruction
                    ax[image_n].imshow(reconstructions, cmap="gray")
                    ax[image_n].set_xticks([])
                    ax[image_n].set_yticks([])
                    ax[image_n].set_ylabel(f"Epoch {epoch:.0f}")

                # Save each figure separately with the epoch number in the filename
                    img_filename = f"reconstruction_epoch_{int(epoch)}.png"
                    plt.imsave(os.path.join(output_dir_images, img_filename), reconstructions, cmap="gray")

            # Save the combined intermediary images plot with a unique timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(output_dir_images, f'intermediary_images_{timestamp}.png'))
            plt.close()  # Close the plot to avoid displaying it

                
    # Finish WandB run
    wandb.finish()
    
def test_model(test_loader):
    
    import os
    import torch

    # Directory to save the test images
    output_dir_test_images = 'test_images'
    os.makedirs(output_dir_test_images, exist_ok=True)

    # Initialize MMD and SSIM metrics
    mmd_scores = []
    mmd = MMDMetric()

    ms_ssim_recon_scores = []
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)

    # Start testing
    print("\nStarting Testing...")
    model.eval()

    # Get a batch of test images
    test_batch = next(iter(test_loader)) 
    images = test_batch["image"].to(device)

    images = images.squeeze(1)

#     # Forward pass through the model to get reconstructions
    with torch.no_grad():
        reconstruction, _ = model(images=images)

    # Compute MMD scores
    for i in range(images.shape[0]):  # Iterate over batch
        mmd_scores.append(mmd(images[i], reconstruction[i]))

    mmd_scores = torch.stack(mmd_scores)
    mmd_mean = mmd_scores.mean().item()
    mmd_std = mmd_scores.std().item()
    print(f"MMD score: {mmd_mean:.4f} ± {mmd_std:.4f}")

    # Compute MSSIM scores
    for i in range(images.shape[0]):  # Iterate over batch
        ms_ssim_recon_scores.append(ms_ssim(images[i], reconstruction[i]))

    mssimscores = torch.stack(ms_ssim_recon_scores)
    ms_ssim_mean = mssimscores.mean().item()
    ms_ssim_std = mssimscores.std().item()
    print(f"MSSIM score: {ms_ssim_mean:.4f} ± {ms_ssim_std:.4f}")

    # Save the MMD and MSSIM scores to a file
    metrics_file = os.path.join(output_dir_test_images, 'test_metrics_1.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"MMD score: {mmd_mean:.4f} ± {mmd_std:.4f}\n")
        f.write(f"MSSIM score: {ms_ssim_mean:.4f} ± {ms_ssim_std:.4f}\n")
        
        
    num_images = 4  # Number of test images to compare

    for i in range(num_images):
        slice_idx = images[i, 0].shape[-1] // 2  # Get middle slice for each test image

        original = images[i, 0, :, :, slice_idx].cpu().numpy()
        recon = reconstruction[i, 0, :, :, slice_idx].cpu().numpy()

        # Create a new figure for each comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot Original
        axes[0].imshow(original, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title(f"Original Test Image {i+1}")

        # Plot Reconstruction
        axes[1].imshow(recon, cmap="gray")
        axes[1].axis("off")
        axes[1].set_title(f"Reconstructed Image {i+1}")

        # Save each figure separately
        image_filename = os.path.join(output_dir_test_images, f'test_image_comparison_{i+1}.png')
        plt.tight_layout()
        plt.savefig(image_filename)
        plt.close()  # Close figure to free memory

    print(f"Saved {num_images} test image comparisons to {output_dir_test_images}")

#     # Select a middle slice for comparison
#     slice_idx = images.shape[-1] // 2  # Middle slice in depth
#     original = images[0, 0, :, :, slice_idx].cpu().numpy()
#     recon = reconstruction[0, 0, :, :, slice_idx].cpu().numpy()

#     # Plot Original vs. Reconstruction
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     axes[0].imshow(original, cmap="gray")
#     axes[0].axis("off")
#     axes[0].set_title("Original Test Image")

#     axes[1].imshow(recon, cmap="gray")
#     axes[1].axis("off")
#     axes[1].set_title("Reconstructed Test Image")

#     # Save the figure and close it to free memory
#     image_filename = os.path.join(output_dir_test_images, 'test_image_comparison_1.png')
#     plt.tight_layout()
#     plt.savefig(image_filename)
#     plt.close()

    
    

if __name__ == "__main__":
    path = "/ifs/loni/faculty/thompson/four_d/ADNI/Spreadsheets_ADNI/ADNI_all_T1_DLpaths_DWIpaths_demographics_20240418_shared.xlsx"
    device = initialize_gpu(5)
    
    train, test, val = load_and_preprocess_data(path)
    
    
    print(len(train))
    print(len(test))
    
    
    print(f"Using {device}")
    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 256),
        # num_res_channels=256,
        num_res_channels=256,
        # num_res_layers=2,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        # upsample_parameters=((2, 4, 1, 1, 1), (2, 4, 1, 1, 1)),
        # num_embeddings = 512,
        # embedding_dim = 128,
        
        num_embeddings = 256,
        embedding_dim = 32,
 
    )
    
    
    mmd = MMDMetric()
    
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
    ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
    
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-5}, 
        {'params': model.decoder.parameters(), 'lr': 1e-4},  
    ], lr=1e-5)  
    
    l1_loss = L1Loss()
    scaler = GradScaler()
    
    
#     train_model(train, val)
    
    
#     test_model(test)
    
    
 
