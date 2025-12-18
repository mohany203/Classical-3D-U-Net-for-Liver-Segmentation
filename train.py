import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import glob
import time
import sys
from tqdm import tqdm

from src.model import ClassicalUNet3D
from src.dataset import LiverResizeDataset
from src.utils import EarlyStopping, calculate_dice, count_parameters

# --- CONFIGURATION ---
DATA_DIR = "./data"  # Path to your dataset folder containing .nii files
SAVE_PATH = "models/best_classical_unet.pth"
LR = 0.0003
EPOCHS = 100
BATCH_SIZE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file_lists(data_dir):
    """
    Scans the data directory for images and labels.
    Assumes a structure where images and labels are identifiable or paired.
    You might need to adjust this pattern matching based on your specific filenames.
    """
    # Example logic based on standard Task03_Liver dataset structures
    # This is a placeholder logic - USER MUST ADAPT TO THEIR DATA PATHS
    # Looking for pair: (volume-X.nii, segmentation-X.nii)
    
    # Recursive search for .nii or .nii.gz
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                all_files.append(os.path.join(root, file))
    
    # Simple logic to separate images and labels. 
    # Usually labels have 'segmentation' or 'label' in name, images might be 'volume'.
    image_paths = sorted([f for f in all_files if 'volume' in f.lower() or 'image' in f.lower()])
    label_paths = sorted([f for f in all_files if 'segmentation' in f.lower() or 'label' in f.lower() or 'mask' in f.lower()])
    
    if len(image_paths) != len(label_paths):
        print(f"Warning: Found {len(image_paths)} images and {len(label_paths)} labels. Check your data directory.")
    
    return image_paths, label_paths

def train():
    # 1. Prepare Data
    print("Preparing Data...")
    image_files, label_files = get_file_lists(DATA_DIR)
    
    if not image_files:
        print(f"No data found in {DATA_DIR}. Please check the path.")
        return

    # Split
    # Using small test/val split as per notebook (100 train, 23 val approx)
    # Adjust test_size as needed.
    train_img_paths, val_img_paths, train_lbl_paths, val_lbl_paths = train_test_split(
        image_files, label_files, 
        test_size=0.2, 
        random_state=12, 
        shuffle=True
    )
    
    train_ds = LiverResizeDataset(train_img_paths, train_lbl_paths, augment=True)
    val_ds = LiverResizeDataset(val_img_paths, val_lbl_paths, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    print("Initializing Model...")
    model = ClassicalUNet3D().to(DEVICE)
    print(f"Model Parameters: {count_parameters(model):,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7)

    # 3. Training Loop
    print(f"Starting Training on {DEVICE}...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_dice = 0
        
        # Train Step
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            d = calculate_dice(outputs, labels)
            train_loss += loss.item()
            train_dice += d.item()
            
            bar.set_postfix(loss=loss.item(), dice=d.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation Step
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_dice += calculate_dice(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Updates
        scheduler.step(avg_val_dice)
        early_stopping(avg_val_dice, avg_val_loss, avg_train_dice, avg_train_loss, model, epoch+1)
        
        # Logging
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Dice:   {avg_val_dice:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    # Save Best Model
    if early_stopping.best_model_state is not None:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(early_stopping.best_model_state, SAVE_PATH)
        print(f"Training Finished. Best Model Saved to: {SAVE_PATH}")
    
    print(f"Total Training Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
