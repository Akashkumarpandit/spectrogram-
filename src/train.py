import os
import sys
import yaml
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import AASIST, AASISTLight, AMSoftmaxLoss
from src.datasets import build_combined_dataset
from src.utils import compute_all_metrics, save_checkpoint, load_checkpoint


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)


def train_one_epoch(
    model: nn.Module,
    loss_module: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 5.0,
    scaler: torch.cuda.amp.GradScaler = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training Batch", leave=False)
    for waveforms, labels, _ in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision context if CUDA
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward pass: extract latent embeddings
            embeddings, _ = model.forward_features(waveforms)
            loss = loss_module(embeddings, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_module.parameters()), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_module.parameters()), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_module: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []

    for waveforms, labels, _ in tqdm(val_loader, desc="Evaluating", leave=False):
        waveforms = waveforms.to(device)
        
        embeddings, _ = model.forward_features(waveforms)
        # In AM-Softmax, bona fide score = cosine similarity with class 1 (bonafide)
        scores = loss_module.get_normalized_scores(embeddings)
        bonafide_scores = scores[:, 1].cpu().numpy()

        all_scores.extend(bonafide_scores)
        all_labels.extend(labels.numpy())

    y_scores = np.array(all_scores)
    y_true = np.array(all_labels)

    metrics = compute_all_metrics(y_true, y_scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AASIST on Combined ASVspoof 2019 & 2021")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    # Set seed
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = get_device(cfg.get("device", "auto"))
    print(f"[*] Running on compute device: {device}")

    # Build Datasets
    ds_cfg = cfg["dataset"]
    print("[*] Parsing combined train protocols...")
    train_dataset = build_combined_dataset(
        protocol_configs=ds_cfg["protocols"]["train"],
        sample_rate=ds_cfg["sample_rate"],
        num_samples=ds_cfg["num_samples"],
        is_train=True,
        pad_mode=ds_cfg["pad_mode"],
        slice_mode=ds_cfg["slice_mode_train"],
        augment=ds_cfg["augmentation"]["enabled"],
        rawboost_algo=ds_cfg["augmentation"]["rawboost"]["algo"],
    )

    print("[*] Parsing combined validation/dev protocols...")
    val_dataset = build_combined_dataset(
        protocol_configs=ds_cfg["protocols"]["dev"],
        sample_rate=ds_cfg["sample_rate"],
        num_samples=ds_cfg["num_samples"],
        is_train=False,
        pad_mode=ds_cfg["pad_mode"],
        slice_mode=ds_cfg["slice_mode_eval"],
        augment=False,
    )

    print(f"[*] Train utterances: {len(train_dataset)} | Dev utterances: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("[!] Warning: Train dataset is empty. Please check paths in config.yaml or run synthetic_data_generator.py first!")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate Model
    m_cfg = cfg["model"]
    arch = m_cfg.get("architecture", "AASIST")
    if arch == "AASIST-L":
        model = AASISTLight(sample_rate=ds_cfg["sample_rate"]).to(device)
    else:
        model = AASIST(sample_rate=ds_cfg["sample_rate"]).to(device)

    # Instantiate Loss (AM-Softmax)
    l_cfg = cfg["loss"]
    loss_module = AMSoftmaxLoss(
        in_features=model.embedding_dim,
        num_classes=2,
        margin=l_cfg.get("margin", 0.2),
        scale=l_cfg.get("scale", 30.0),
        class_weights=l_cfg.get("class_weights", [0.1, 0.9]),
    ).to(device)

    # Optimizer & Scheduler
    params = list(model.parameters()) + list(loss_module.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_decay_step"],
        gamma=cfg["training"]["lr_decay_gamma"],
    )

    # Mixed precision scaler (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if (cfg["training"]["mixed_precision"] and device.type == "cuda") else None

    start_epoch = 1
    best_eer = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    if args.resume:
        print(f"[*] Resuming from: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, loss_module, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float("inf"))

    total_epochs = cfg["training"]["epochs"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]

    print(f"\n{'='*60}")
    print(f"🚀 Starting AASIST Training ({start_epoch} -> {total_epochs} epochs)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(model, loss_module, train_loader, optimizer, device, cfg["training"]["gradient_clip"], scaler)
        scheduler.step()

        print(f"[Epoch {epoch:03d}/{total_epochs:03d}] Train Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation
        if len(val_dataset) > 0 and (epoch % cfg["training"].get("eval_frequency", 1) == 0):
            val_metrics = evaluate(model, loss_module, val_loader, device)
            val_eer = val_metrics["eer"]
            print(f"  └── Validation -> EER: {val_eer:.2f}% | min t-DCF: {val_metrics['min_tdcf']:.4f} | Accuracy: {val_metrics['accuracy']:.2f}% | AUC: {val_metrics['auc']:.2f}%")

            is_best = val_eer < best_eer
            if is_best:
                best_eer = val_eer
                patience_counter = 0
                print(f"  ⭐ New Best Model! (EER: {best_eer:.2f}%)")
            else:
                patience_counter += 1

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": loss_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eer": best_eer,
                    "metrics": val_metrics,
                    "config": cfg,
                },
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                is_best=is_best,
            )

            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered after {patience} epochs without improvement.")
                break


if __name__ == "__main__":
    main()
