import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    epoch: int,
    is_best: bool = False,
    filename: Optional[str] = None,
) -> str:
    """Save model checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_module: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint and restore weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Load loss module (AM-Softmax) weights if available
    if loss_module is not None and "loss_state_dict" in checkpoint:
        loss_module.load_state_dict(checkpoint["loss_state_dict"], strict=False)

    # Load optimizer state if resuming training
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
