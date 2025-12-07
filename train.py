import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import PolypDataset
from src.model import UNet
import os

# --- CONFIGURATION ---
IMG_DIR = "data/Kvasir-SEG/images"
MASK_DIR = "data/Kvasir-SEG/masks"
LR = 1e-4
BATCH_SIZE = 16  # Increased to 16 since images are smaller (128x128)
EPOCHS = 3     
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Training on: {DEVICE}")
    
    # 1. Load Data
    full_dataset = PolypDataset(IMG_DIR, MASK_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    model = UNet().to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Add a counter to see progress within the epoch
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print every 10 batches to show it's alive
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}...")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), "unet_polyp_128.pth")
    print("Training Complete. Model saved as unet_polyp_128.pth")

if __name__ == "__main__":
    if not os.path.exists(IMG_DIR):
        print("Data not found! Please run 'python download_data.py' first.")
    else:
        train()