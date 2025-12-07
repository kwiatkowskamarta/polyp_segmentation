import torch
import cv2
import os
import matplotlib.pyplot as plt
from src.dataset import PolypDataset
from src.model import UNet
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
# Use absolute paths to be safe
BASE_DIR = os.getcwd()
IMG_DIR = os.path.join(BASE_DIR, "data", "Kvasir-SEG", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "Kvasir-SEG", "masks")
MODEL_PATH = "unet_polyp_128.pth"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_paths():
    print(f"Current Working Directory: {BASE_DIR}")
    print(f"Looking for images in: {IMG_DIR}")
    
    if not os.path.exists(IMG_DIR):
        print("ERROR: Image directory not found!")
        return False
    
    num_images = len(os.listdir(IMG_DIR))
    print(f"Found {num_images} images.")
    
    if num_images == 0:
        print("ERROR: Image directory is empty.")
        return False
        
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Did you run train.py?")
        return False
        
    return True

def calculate_dice(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate():
    if not check_paths():
        return

    # Create output dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Loading model from {MODEL_PATH}...")
    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    # Load Data
    full_dataset = PolypDataset(IMG_DIR, MASK_DIR)
    total_samples = len(full_dataset)
    print(f"Total Dataset Size: {total_samples}")
    
    # Split
    train_size = int(0.8 * total_samples)
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_samples))
    print(f"Validation Set Size: {len(val_dataset)}")
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    dice_scores = []
    images_saved = 0
    
    print("Starting Inference Loop...")
    
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            output = model(data)
            pred_mask = (output > 0.5).float()
            
            score = calculate_dice(pred_mask, target)
            dice_scores.append(score)
            
            # Save first 5 images
            if images_saved < 5:
                print(f" Saving image {images_saved+1}/5 (Dice: {score:.2f})...")
                
                # Undo normalization for plotting
                img_np = data[0].cpu().permute(1, 2, 0).numpy()
                target_np = target[0].cpu().squeeze().numpy()
                pred_np = pred_mask[0].cpu().squeeze().numpy()
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.title("Input"); plt.imshow(img_np); plt.axis("off")
                plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(target_np, cmap="gray"); plt.axis("off")
                plt.subplot(1, 3, 3); plt.title(f"Prediction (Dice: {score:.2f})"); plt.imshow(pred_np, cmap="gray"); plt.axis("off")
                
                save_path = os.path.join(OUTPUT_DIR, f"result_{images_saved}.png")
                plt.savefig(save_path)
                plt.close()
                images_saved += 1
    
    if len(dice_scores) > 0:
        avg_dice = sum(dice_scores) / len(dice_scores)
        print(f"\nEvaluation Complete!")
        print(f"Average Dice Score: {avg_dice:.4f}")
        print(f" Check the folder '{os.path.abspath(OUTPUT_DIR)}' for results.")
    else:
        print("Warning: No validation data was processed.")

if __name__ == "__main__":
    evaluate()