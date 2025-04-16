from seg_model import binary_seg_model
from cyto_utils import get_train_transforms, get_val_transforms
from cyto_dataset import CytologyDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from loss import CombinedLoss, DiceLoss

device = 'cuda:4'

train_ds = CytologyDataset(geojson_dir='geojsons/', mode='train', transform=get_train_transforms())
val_ds = CytologyDataset(geojson_dir='geojsons/', mode='val', transform=get_val_transforms())
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

model = binary_seg_model().to(device)
criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    train_combined_loss = 0.0
    train_bce_loss = 0.0
    train_dice_loss = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)  # shape: [B, 1, H, W]
        
        outputs = model(images)
        combined_loss, bce_loss, dice_loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        combined_loss.backward()  # Backprop with combined loss
        optimizer.step()
        
        train_combined_loss += combined_loss.item()
        train_bce_loss += bce_loss.item()
        train_dice_loss += dice_loss.item()
    
    avg_train_combined = train_combined_loss / len(train_loader)
    avg_train_bce = train_bce_loss / len(train_loader)
    avg_train_dice = train_dice_loss / len(train_loader)
    
    model.eval()
    val_bce_loss = 0.0
    val_dice_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            
            bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
            dice_loss = DiceLoss()(outputs, masks)
            
            val_bce_loss += bce_loss.item()
            val_dice_loss += dice_loss.item()
    
    avg_val_bce = val_bce_loss / len(val_loader)
    avg_val_dice = val_dice_loss / len(val_loader)
    
    print(f"Epoch {epoch+1:04d}/{num_epochs}: "
          f"Train Combined: {avg_train_combined:.4f}, "
          f"Train BCE: {avg_train_bce:.4f}, "
          f"Train Dice: {avg_train_dice:.4f} | "
          f"Val BCE: {avg_val_bce:.4f}, "
          f"Val Dice: {avg_val_dice:.4f}")