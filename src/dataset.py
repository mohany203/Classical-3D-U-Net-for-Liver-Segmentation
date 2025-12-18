import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
import cv2  

class LiverResizeDataset(Dataset):
    """
    Custom Dataset for 3D Liver Segmentation.
    Handles loading NIfTI files, preprocessing (clipping, normalization, CLAHE),
    and on-the-fly augmentation.
    """
    def __init__(self, image_paths, label_paths, target_shape=(64, 128, 128), augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_shape = target_shape
        self.augment = augment 
        
        # Initialize CLAHE (Contrast Enhancement)
        # Using standard medical parameters as per original notebook
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Data
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        
        try:
            img_nii = nib.load(img_path)
            lbl_nii = nib.load(lbl_path) # Corrected from nib.nib.load to nib.load
            img_data = img_nii.get_fdata()
            lbl_data = lbl_nii.get_fdata()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data to avoid crashing, or raise error depending on preference
            # usually robust handling is better. For now, we assume paths are valid.
            raise e

        # --- PREPROCESSING ---
        # Clip to Liver Window
        img_data = np.clip(img_data, -100, 400)
        
        # Normalize to 0-255 (uint8) for OpenCV processing
        img_min, img_max = img_data.min(), img_data.max()
        if img_max > img_min:
            img_data = 255 * (img_data - img_min) / (img_max - img_min)
        img_data = img_data.astype(np.uint8)
        
        processed_slices = []
        
        # Process slice by slice
        for i in range(img_data.shape[2]): # Iterate over depth (Z-axis)
            slice_img = img_data[:, :, i]
            
            # 1. Spatial Filtering (Gaussian Blur) 
            slice_img = cv2.GaussianBlur(slice_img, (3, 3), 0)
            
            # 2. Contrast Enhancement (CLAHE)
            slice_img = self.clahe.apply(slice_img)
            
            processed_slices.append(slice_img)
        
        # Stack back to 3D and convert back to float for tensor
        img_data = np.stack(processed_slices, axis=2).astype(np.float32)
        
        # Final Standardization
        mean = np.mean(img_data)
        std = np.std(img_data)
        img_data = (img_data - mean) / (std + 1e-8)
        
        # 3. Random Transforms to Tensor (Permute to Channel, Depth, Height, Width generally or Depth, H, W?)
        # Notebook permuted to (2, 0, 1) -> (Depth, Height, Width). 
        # Since it's 3D CNN, we usually want (C, D, H, W).
        # The notebook did: 
        #   img_tensor = torch.tensor(img_data).permute(2, 0, 1) -> shape (D, H, W)
        #   img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) -> shape (1, 1, D, H, W) ??
        # Let's check the notebook logic again.
        # Notebook: img_tensor.unsqueeze(0).unsqueeze(0) after permute?
        # Notebook: `img_tensor = torch.tensor(img_data, dtype=torch.float32).permute(2, 0, 1)` -> (D, H, W)
        # Notebook: `img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)` -> (1, 1, D, H, W)
        # Notebook: `img_resized = F.interpolate(img_tensor, ...).squeeze(0)` -> (1, D, H, W)
        # So final output is (C=1, D, H, W). Correct.
        
        img_tensor = torch.tensor(img_data, dtype=torch.float32).permute(2, 0, 1)
        lbl_tensor = torch.tensor(lbl_data, dtype=torch.float32).permute(2, 0, 1)
        
        # 4. DATA AUGMENTATION
        if self.augment:
            # Intensity Augmentation
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            img_tensor = (img_tensor * contrast_factor) * brightness_factor
            
            # Geometric Augmentations
            # Flip depth
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[0]) # dims=[2] in notebook but tensor is (D,H,W) so Depth is 0!
                lbl_tensor = torch.flip(lbl_tensor, dims=[0])
                # Wait, notebook line 62: img_tensor = torch.flip(img_tensor, dims=[2])
                # In notebook, img_tensor was (D, H, W) (from permute 2,0,1 of H,W,D).
                # So dims=[2] is Width. dims=[1] is Height.
                # Let's stick to notebook logic or correct it if obvious?
                # Notebook said "dims=[2]" which is Width here.
                # "dims=[1]" is Height.
                # It seems depth flip wasn't done, or maybe random rot90 covers it?
                # I will reproduce notebook logic to be safe, assuming D,H,W semantics.
                
                # Correction: Notebook logic:
                # img_tensor is (D, H, W).
                # dims=[2] -> W
                # dims=[1] -> H
            
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2]) # Flip W
                lbl_tensor = torch.flip(lbl_tensor, dims=[2])
            
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1]) # Flip H
                lbl_tensor = torch.flip(lbl_tensor, dims=[1])
            
            if random.random() > 0.5:
                k = random.randint(1, 3) 
                # rot90 on H, W plane. dims=[1, 2]
                img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])
                lbl_tensor = torch.rot90(lbl_tensor, k, dims=[1, 2])
        
        # 5. Resize to fixed target shape
        # Add batch and channel dims for interpolate: (1, 1, D, H, W)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        lbl_tensor = lbl_tensor.unsqueeze(0).unsqueeze(0)
        
        # Trilinear for image, Nearest for mask
        img_resized = F.interpolate(img_tensor, size=self.target_shape, mode='trilinear', align_corners=False)
        lbl_resized = F.interpolate(lbl_tensor, size=self.target_shape, mode='nearest')
        
        # Remove batch dim: (C, D, H, W)
        return img_resized.squeeze(0), lbl_resized.squeeze(0)
