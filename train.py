"""
Training script for U-Net cyclone segmentation model.
Loads pre-split data from raw Zarr stores with CUDA memory safety and DataParallel support.
"""

import os
import gc
import zarr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import config
from model import U_Net, Trainer, init_weights


# =============================================================================
# Dataset Class (Thread-safe Lazy Zarr Opening)
# =============================================================================
class CycloneZarrDataset(Dataset):
    """
    Direct Zarr dataset loader opening file handles inside worker processes.
    """
    def __init__(self, zarr_group_path):
        super().__init__()
        self.zarr_group_path = zarr_group_path
        self.root = None

    def _open_zarr(self):
        if self.root is None:
            self.root = zarr.open_group(self.zarr_group_path, mode='r')
            self.inputs = self.root['inputs']
            self.labels = self.root['labels']

    def __len__(self):
        if self.root is None:
            self._open_zarr()
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        self._open_zarr()

        x_np = np.asarray(self.inputs[idx], dtype=np.float32)
        y_np = np.asarray(self.labels[idx], dtype=np.float32)

        x = torch.from_numpy(x_np).permute(2, 0, 1)
        y = torch.from_numpy(y_np).permute(2, 0, 1)

        return x, y


# =============================================================================
# Set CUDA Memory Allocation Configuration
# =============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# Micro-batch size per forward pass
MICRO_BATCH_SIZE = getattr(config, 'micro_batch_size', 8)
ACCUMULATION_STEPS = max(1, config.batch_size // MICRO_BATCH_SIZE)

print(f"Target Effective Batch Size: {config.batch_size}")
print(f"Micro Batch Size: {MICRO_BATCH_SIZE} | Accumulation Steps: {ACCUMULATION_STEPS}")


# =============================================================================
# Load Data & Construct Datasets
# =============================================================================
print("\nOpening train, valid, test Zarr groups...")
train_zarr_path = os.path.join(config.data_dir, 'train_data.zarr')
valid_zarr_path = os.path.join(config.data_dir, 'valid_data.zarr')
test_zarr_path = os.path.join(config.data_dir, 'test_data.zarr')

train_dataset = CycloneZarrDataset(train_zarr_path)
valid_dataset = CycloneZarrDataset(valid_zarr_path)
test_dataset = CycloneZarrDataset(test_zarr_path)

print(f"Train dataset samples: {len(train_dataset)} | Input shape: {train_dataset.inputs.shape}")
print(f"Valid dataset samples: {len(valid_dataset)} | Input shape: {valid_dataset.inputs.shape}")
print(f"Test dataset samples:  {len(test_dataset)}  | Input shape: {test_dataset.inputs.shape}")

use_workers = config.num_workers > 0
train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=config.num_workers,
    persistent_workers=use_workers,
    prefetch_factor=2 if use_workers else None,
    pin_memory=torch.cuda.is_available()
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=False,
    num_workers=config.num_workers,
    persistent_workers=use_workers,
    prefetch_factor=4 if use_workers else None,
    pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=False,
    num_workers=config.num_workers,
    persistent_workers=use_workers,
    prefetch_factor=2 if use_workers else None,
    pin_memory=torch.cuda.is_available()
)


# =============================================================================
# Initialize Model & Multi-GPU Parallelization (nn.DataParallel)
# =============================================================================
load_path = config.model_detect_path
try:
    print(f"Loading pretrained model from {load_path}...")
    jit_module = torch.jit.load(load_path, map_location=config.device)
    base_model = U_Net()
    base_model.load_state_dict(jit_module.state_dict())
    base_model.to(config.device)
except:
    print("\nLoading model failed...")
    print("\nInitializing new model...")
    base_model = U_Net()
    base_model.apply(init_weights)
    base_model.to(config.device)


# Wrap model with DataParallel if multiple GPUs are available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = nn.DataParallel(base_model)

else:
    print(f"Running model on single device: {config.device}")
    model = base_model

optimizer = config.optimiser(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler('cuda', enabled=torch.cuda.is_available())


# =============================================================================
# Train Loop with Gradient Accumulation & Automatic Mixed Precision (AMP)
# =============================================================================
print(f"\nTraining for {config.epochs} epochs...")

for epoch in range(1, config.epochs + 1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(config.device, non_blocking=True)
        targets = targets.to(config.device, non_blocking=True)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * ACCUMULATION_STEPS

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{config.epochs}] - Train Loss: {epoch_loss:.6f}")

    torch.cuda.empty_cache()
    gc.collect()

print("\nTraining complete!")


# =============================================================================
# Test Evaluation
# =============================================================================
print("\nEvaluating on test set...")
model.eval()
test_losses = []
correct_pixels = 0
total_pixels = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(config.device, non_blocking=True)
        targets = targets.to(config.device, non_blocking=True)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        test_losses.append(loss.item())

        pred_classes = torch.argmax(outputs, dim=1)
        target_classes = torch.argmax(targets, dim=1)

        correct_pixels += (pred_classes == target_classes).sum().item()
        total_pixels += target_classes.numel()

mean_test_loss = np.mean(test_losses)
accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0

print(f"Test Loss:     {mean_test_loss:.6f}")
print(f"Test Accuracy: {accuracy:.4f}")


# =============================================================================
# Save Model Weights
# =============================================================================
save_path = config.model_detect_path
print(f"\nSaving model to {save_path}...")

model_to_save = model.module if isinstance(model, nn.DataParallel) else model
model_to_save.eval()

sample_inputs, _ = test_dataset[0]
lat, lon, x_var = sample_inputs.shape
dummy_input = torch.randn(1, lat, lon, x_var, device=config.device)
 
with torch.no_grad():
    traced_model = torch.jit.trace(model_to_save, dummy_input)

traced_model.save(save_path)
print("TorchScript JIT model saved successfully!")