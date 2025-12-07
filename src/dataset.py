import os
import cv2
import torch
from torch.utils.data import Dataset

class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. Load Image
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        # Read as RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Mask as Grayscale
        mask = cv2.imread(mask_path, 0)

        # --- CHANGE IS HERE ---
        # 2. Resize to 128x128 for faster CPU training
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        # ----------------------

        # 3. Normalize & Convert to Tensor
        image = image / 255.0
        image = image.transpose((2, 0, 1)) 
        image = torch.tensor(image, dtype=torch.float32)

        mask = mask / 255.0
        mask = (mask > 0.5).astype(float)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask