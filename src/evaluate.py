import os
import sys
import yaml
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import AASIST, AASISTLight, AMSoftmaxLoss
from src.datasets import build_combined_dataset
from src.utils import compute_all_metrics, compute_eer, load_checkpoint
from src.train import get_device


def evaluate_protocol(
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    output_score_file: str = "eval_scores.txt",
) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()

    utt_ids_list: List[str] = []
    scores_list: List[float] = []
    labels_list: List[int] = []
    attacks_list: List[str] = []

    print(f"[*] Scoring {len(dataset)} evaluation utterances...")
    with torch.no_grad():
        for i, (waveforms, labels, utt_ids) in enumerate(tqdm(loader, desc="Evaluation")):
            waveforms = waveforms.to(device)
            embeddings, _ = model.forward_features(waveforms)
            scores = loss_module.get_normalized_scores(embeddings)
            
            # Bonafide score (class 1)
            bona_scores = scores[:, 1].cpu().numpy()

            utt_ids_list.extend(utt_ids)
            scores_list.extend(bona_scores)
            labels_list.extend(labels.numpy())

            # Retrieve attack types from dataset items
            batch_start = i * batch_size
            for idx in range(len(utt_ids)):
                item_idx = batch_start + idx
                if item_idx < len(dataset.items):
                    attacks_list.append(dataset.items[item_idx].attack_type)
                else:
                    attacks_list.append("-")

    y_scores = np.array(scores_list)
    y_true = np.array(labels_list)

    # Save scores file in standard ASVspoof format
    os.makedirs(os.path.dirname(output_score_file) or ".", exist_ok=True)
    with open(output_score_file, "w", encoding="utf-8") as f:
        for uid, sc, lb, atk in zip(utt_ids_list, scores_list, labels_list, attacks_list):
            tag = "bonafide" if lb == 1 else "spoof"
            f.write(f"{uid} {atk} {tag} {sc:.6f}\n")

    print(f"[*] Scores successfully written to: {output_score_file}")

    # Compute global metrics
    global_metrics = compute_all_metrics(y_true, y_scores)

    # Compute breakdown per attack type
    unique_attacks = sorted(list(set(attacks_list)))
    bonafide_mask = (y_true == 1)
    bonafide_scores = y_scores[bonafide_mask]

    attack_breakdown = {}
    for atk in unique_attacks:
        if atk == "-" or atk == "bonafide":
            continue
        atk_mask = np.array([a == atk for a in attacks_list])
        if np.sum(atk_mask) > 0:
            atk_spoof_scores = y_scores[atk_mask]
            atk_eer, _ = compute_eer(bonafide_scores, atk_spoof_scores)
            attack_breakdown[atk] = {
                "count": int(np.sum(atk_mask)),
                "eer": round(atk_eer, 3),
            }

    return {
        "global": global_metrics,
        "attacks": attack_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AASIST on ASVspoof 2019/2021 protocols")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to master config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--protocol", type=str, default=None, help="Specific protocol file (overrides config)")
    parser.add_argument("--audio_dir", type=str, default=None, help="Audio directory for protocol")
    parser.add_argument("--output_file", type=str, default="evaluation_results/scores.txt", help="Output score file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    print(f"[*] Running on device: {device}")

    # Initialize model
    arch = cfg["model"].get("architecture", "AASIST")
    if arch == "AASIST-L":
        model = AASISTLight().to(device)
    else:
        model = AASIST().to(device)

    loss_module = AMSoftmaxLoss(in_features=model.embedding_dim, num_classes=2).to(device)

    print(f"[*] Loading checkpoint weights: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, loss_module=loss_module, device=str(device))

    # Determine protocol to evaluate
    if args.protocol and args.audio_dir:
        eval_protocols = [{"path": args.protocol, "audio_dir": args.audio_dir, "dataset_version": "2021"}]
    else:
        eval_protocols = cfg["dataset"]["protocols"].get("eval", cfg["dataset"]["protocols"]["dev"])

    eval_dataset = build_combined_dataset(
        protocol_configs=eval_protocols,
        sample_rate=cfg["dataset"]["sample_rate"],
        num_samples=cfg["dataset"]["num_samples"],
        is_train=False,
        pad_mode="repeat",
        slice_mode="center",
    )

    if len(eval_dataset) == 0:
        print("[!] Evaluation dataset is empty. Check protocol paths.")
        sys.exit(1)

    results = evaluate_protocol(
        model=model,
        loss_module=loss_module,
        dataset=eval_dataset,
        device=device,
        batch_size=cfg["training"]["batch_size"],
        output_score_file=args.output_file,
    )

    gm = results["global"]
    print(f"\n{'='*60}")
    print("🎯 EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Equal Error Rate (EER)   : {gm['eer']:.3f}%")
    print(f"Min tandem DCF (min t-DCF): {gm['min_tdcf']:.4f}")
    print(f"Optimal Threshold        : {gm['threshold']:.4f}")
    print(f"ROC-AUC                  : {gm['auc']:.2f}%")
    print(f"Accuracy                 : {gm['accuracy']:.2f}%")
    print(f"Precision / Recall / F1  : {gm['precision']:.2f}% / {gm['recall']:.2f}% / {gm['f1']:.2f}%")

    if results["attacks"]:
        print(f"\n{'-'*60}")
        print("ATTACK BREAKDOWN (Per-Attack EER)")
        print(f"{'-'*60}")
        for atk, data in results["attacks"].items():
            print(f"  Attack {atk:10s} | Utterances: {data['count']:5d} | EER: {data['eer']:6.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
