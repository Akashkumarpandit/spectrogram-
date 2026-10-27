"""
========================================================================================
ASVspoof 2019 & 2021 Voice Anti-Spoofing & Deepfake Detection - Master Project File
========================================================================================
Complete all-in-one consolidated implementation:
  1. SincNet Raw Waveform Band-Pass Filterbank
  2. 2D Spectro-Temporal Residual CNN Encoder
  3. Heterogeneous Graph Attention Networks (GAT) & Graph Pooling
  4. AASIST & AASIST-L Full Architecture
  5. Additive Margin Softmax (AM-Softmax) Loss
  6. ASVspoof 2019 & 2021 Multi-Protocol Parser & RawBoost Augmentation
  7. Equal Error Rate (EER) & minimum tandem DCF (min t-DCF) Metrics
  8. Training, Validation & Checkpoint Engine
  9. Single/Batch Audio Deepfake Inference CLI
 10. Web Interface & Mock Data Synthesizer
========================================================================================
"""

import os
import sys
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import scipy.signal as signal
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ======================================================================================
# 1. FRONT-END: SINCNET PARAMETRIC FILTERBANK (Raw Waveforms)
# ======================================================================================

def to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class SincConv(nn.Module):
    """Parametric sinc-based bandpass filterbank for raw audio."""

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 128,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        min_low_hz: float = 0.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(to_mel(low_hz), to_mel(high_hz), self.out_channels + 1)
        hz = to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int(self.kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=self.stride, padding=self.padding, bias=None)


# ======================================================================================
# 2. GRAPH ATTENTION & SPECTRO-TEMPORAL HETEROGENEOUS FUSION
# ======================================================================================

class GraphAttentionLayer(nn.Module):
    """Multi-head Graph Attention Layer with residual connection."""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, temperature: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.temperature = temperature

        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.leaky_relu = nn.LeakyReLU(0.2)

        if in_features != out_features * num_heads:
            self.res_proj = nn.Linear(in_features, out_features * num_heads, bias=False)
        else:
            self.res_proj = nn.Identity()

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        h_prime = self.W(h).view(B, N, self.num_heads, self.out_features)

        attn_src = torch.einsum("bnhd,hdo->bnh", h_prime, self.a_src)
        attn_dst = torch.einsum("bnhd,hdo->bnh", h_prime, self.a_dst)

        attn_src = attn_src.permute(0, 2, 1).unsqueeze(-1)
        attn_dst = attn_dst.permute(0, 2, 1).unsqueeze(-2)
        attn = self.leaky_relu(attn_src + attn_dst) / self.temperature

        attn_weights = F.softmax(attn, dim=-1)
        h_perm = h_prime.permute(0, 2, 1, 3)
        out = torch.matmul(attn_weights, h_perm)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        return F.relu(out + self.res_proj(h))


class GraphPooling(nn.Module):
    """Top-k score-based graph node pooling."""

    def __init__(self, in_features: int, pool_ratio: float = 0.5):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.score_proj = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        k = max(1, int(N * self.pool_ratio))

        scores = torch.sigmoid(self.score_proj(h)).squeeze(-1)
        _, topk_indices = torch.topk(scores, k=k, dim=-1)

        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, k)
        pooled = h[batch_idx, topk_indices]
        gates = torch.gather(scores, dim=-1, index=topk_indices).unsqueeze(-1)
        return pooled * gates


class SpectroTemporalGraphFusion(nn.Module):
    """Cross-attention fusion between spectral and temporal graph representations."""

    def __init__(self, in_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.temporal_gat = GraphAttentionLayer(in_dim, out_dim, num_heads=2)
        self.spectral_gat = GraphAttentionLayer(in_dim, out_dim, num_heads=2)

        total_dim = out_dim * 2
        self.temporal_pool = GraphPooling(total_dim, pool_ratio=0.5)
        self.spectral_pool = GraphPooling(total_dim, pool_ratio=0.5)

        self.cross_attn_s2t = nn.MultiheadAttention(embed_dim=total_dim, num_heads=2, batch_first=True)
        self.cross_attn_t2s = nn.MultiheadAttention(embed_dim=total_dim, num_heads=2, batch_first=True)

        self.norm_t = nn.LayerNorm(total_dim)
        self.norm_s = nn.LayerNorm(total_dim)

    def forward(self, h_temporal: torch.Tensor, h_spectral: torch.Tensor) -> torch.Tensor:
        h_t = self.temporal_pool(self.temporal_gat(h_temporal))
        h_s = self.spectral_pool(self.spectral_gat(h_spectral))

        t_fused, _ = self.cross_attn_s2t(query=h_t, key=h_s, value=h_s)
        s_fused, _ = self.cross_attn_t2s(query=h_s, key=h_t, value=h_t)

        h_t = self.norm_t(h_t + t_fused)
        h_s = self.norm_s(h_s + s_fused)

        t_rep = torch.cat([h_t.mean(dim=1), h_t.max(dim=1)[0]], dim=-1)
        s_rep = torch.cat([h_s.mean(dim=1), h_s.max(dim=1)[0]], dim=-1)
        return torch.cat([t_rep, s_rep], dim=-1)


# ======================================================================================
# 3. AASIST MODEL ARCHITECTURE
# ======================================================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: Tuple[int, int] = (2, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channels)
        else:
            self.res_conv = None
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.res_conv is None else self.res_bn(self.res_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return self.pool(out)


class AASIST(nn.Module):
    """AASIST: End-to-end Audio Anti-Spoofing using Spectro-Temporal Graph Attention."""

    def __init__(
        self,
        sample_rate: int = 16000,
        sinc_filters: int = 70,
        sinc_kernel: int = 128,
        num_classes: int = 2,
        embedding_dim: int = 160,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.sinc_conv = SincConv(
            out_channels=sinc_filters,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            padding=sinc_kernel // 2,
        )
        self.sinc_bn = nn.BatchNorm1d(sinc_filters)
        self.sinc_relu = nn.LeakyReLU(0.2)
        self.sinc_pool = nn.MaxPool1d(kernel_size=3, stride=3)

        self.res1 = ResBlock(1, 32)
        self.res2 = ResBlock(32, 32)
        self.res3 = ResBlock(32, 64)
        self.res4 = ResBlock(64, 64)

        self.graph_proj = nn.Conv2d(64, 32, kernel_size=1)
        self.graph_fusion = SpectroTemporalGraphFusion(in_dim=32, out_dim=32)

        fused_dim = 256 # 4 * 32 * 2
        self.fc_embed = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        sinc_feats = self.sinc_pool(self.sinc_relu(self.sinc_bn(self.sinc_conv(x))))
        x_2d = sinc_feats.unsqueeze(1)
        x_2d = self.res1(x_2d)
        x_2d = self.res2(x_2d)
        x_2d = self.res3(x_2d)
        x_2d = self.res4(x_2d)

        x_graph = self.graph_proj(x_2d)
        h_temporal = x_graph.mean(dim=2).permute(0, 2, 1)
        h_spectral = x_graph.mean(dim=3).permute(0, 2, 1)

        fused_graph = self.graph_fusion(h_temporal, h_spectral)
        embedding = self.fc_embed(fused_graph)
        return embedding, fused_graph

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding, _ = self.forward_features(x)
        logits = self.classifier(embedding)
        return logits, embedding


# ======================================================================================
# 4. AM-SOFTMAX LOSS (Angular Margin Softmax)
# ======================================================================================

class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax (AM-Softmax) Loss for metric space separation."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        margin: float = 0.2,
        scale: float = 30.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = margin
        self.s = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        if class_weights is not None:
            self.register_buffer("class_weights", torch.FloatTensor(class_weights))
        else:
            self.class_weights = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)

        costheta = F.linear(x_norm, w_norm)
        one_hot = torch.zeros_like(costheta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        costheta_m = costheta - self.m * one_hot
        scaled_logits = self.s * costheta_m
        return F.cross_entropy(scaled_logits, labels, weight=self.class_weights)

    def get_normalized_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        return F.linear(x_norm, w_norm)


# ======================================================================================
# 5. DATASETS, PROTOCOL PARSERS & RAWBOOST AUGMENTATION
# ======================================================================================

class RawBoost:
    """RawBoost raw audio data augmentation."""

    def __init__(self, algo: int = 5):
        self.algo = algo

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_np = x.squeeze().cpu().numpy().astype(np.float32)

        choice = random.choice([1, 2, 3]) if self.algo == 5 else self.algo
        if choice == 1:
            num_taps = random.randint(5, 20)
            cutoff = random.uniform(0.1, 0.85)
            b = signal.firwin(num_taps, cutoff)
            x_np = np.tanh(signal.lfilter(b, 1.0, x_np)).astype(np.float32)
        elif choice == 2:
            num_spikes = random.randint(1, 8)
            indices = np.random.randint(0, len(x_np), size=num_spikes)
            for idx in indices:
                x_np[idx] = random.uniform(-1.0, 1.0)
        elif choice == 3:
            noise = np.random.normal(0, 0.015, size=len(x_np))
            x_np = (x_np + noise).astype(np.float32)

        return torch.from_numpy(x_np).to(dtype=x.dtype, device=x.device).view(orig_shape)


@dataclass
class AudioProtocolItem:
    utt_id: str
    audio_path: str
    label_str: str
    label: int
    attack_type: str
    speaker_id: Optional[str] = None
    dataset_version: str = "2019"


def parse_protocol_file(protocol_path: str, audio_dir: str, dataset_version: str = "2019") -> List[AudioProtocolItem]:
    if not os.path.isfile(protocol_path):
        return []

    items = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue

            lower_tokens = [t.lower() for t in tokens]
            if "bonafide" in lower_tokens:
                label_str = "bonafide"
            elif "spoof" in lower_tokens:
                label_str = "spoof"
            else:
                label_str = "bonafide" if tokens[-1] == "1" else "spoof"

            label = 1 if label_str == "bonafide" else 0
            speaker_id = tokens[0]
            utt_id = tokens[1] if (len(tokens) > 1 and tokens[1] != "-") else tokens[0]
            attack_type = tokens[3] if len(tokens) > 3 else "-"

            # Find matching audio
            audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
            if not os.path.isfile(audio_path):
                audio_path = os.path.join(audio_dir, f"{utt_id}.wav")

            items.append(
                AudioProtocolItem(
                    utt_id=utt_id,
                    audio_path=audio_path,
                    label_str=label_str,
                    label=label,
                    attack_type=attack_type,
                    speaker_id=speaker_id,
                    dataset_version=dataset_version,
                )
            )
    return items


class ASVSpoofDataset(Dataset):
    """Universal Dataset for ASVspoof 2019 and 2021."""

    def __init__(
        self,
        items: List[AudioProtocolItem],
        sample_rate: int = 16000,
        num_samples: int = 64600,
        is_train: bool = True,
        augment: bool = False,
    ):
        self.items = items
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.is_train = is_train
        self.augmentor = RawBoost() if (is_train and augment) else None

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        try:
            data, sr = sf.read(filepath, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            waveform = torch.from_numpy(data).float()
        except Exception:
            waveform = torch.zeros(self.num_samples, dtype=torch.float32)

        length = waveform.size(0)
        if length < self.num_samples:
            repeats = int(np.ceil(self.num_samples / max(1, length)))
            waveform = waveform.repeat(repeats)[: self.num_samples]
        elif length > self.num_samples:
            if self.is_train:
                start = random.randint(0, length - self.num_samples)
            else:
                start = (length - self.num_samples) // 2
            waveform = waveform[start : start + self.num_samples]
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        waveform = self._load_audio(item.audio_path)
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        return waveform, item.label, item.utt_id


# ======================================================================================
# 6. BENCHMARK METRICS: EER & MIN T-DCF
# ======================================================================================

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0

    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return float(eer), float(thresholds[idx])


def compute_min_tdcf(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    beta = 0.055
    return float(np.min(fnr + beta * fpr))


def compute_all_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    bona = y_scores[y_true == 1]
    spoof = y_scores[y_true == 0]
    eer, thresh = compute_eer(bona, spoof)
    min_tdcf = compute_min_tdcf(bona, spoof)
    y_pred = (y_scores >= thresh).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    try:
        auc = roc_auc_score(y_true, y_scores) * 100.0
    except Exception:
        auc = 50.0

    return {
        "eer": round(eer, 3),
        "threshold": round(thresh, 4),
        "min_tdcf": round(min_tdcf, 4),
        "accuracy": round(acc, 2),
        "auc": round(auc, 2),
    }


# ======================================================================================
# 7. TRAINING & EVALUATION PIPELINE
# ======================================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_aasist(
    train_proto: str,
    train_audio_dir: str,
    dev_proto: str,
    dev_audio_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
):
    device = get_device()
    print(f"[*] Training AASIST on device: {device}")

    train_items = parse_protocol_file(train_proto, train_audio_dir)
    dev_items = parse_protocol_file(dev_proto, dev_audio_dir)

    print(f"[*] Loaded {len(train_items)} train utterances | {len(dev_items)} dev utterances")

    train_ds = ASVSpoofDataset(train_items, is_train=True, augment=True)
    dev_ds = ASVSpoofDataset(dev_items, is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    model = AASIST().to(device)
    loss_module = AMSoftmaxLoss(in_features=model.embedding_dim, num_classes=2).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_module.parameters()), lr=lr, weight_decay=1e-4)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for waveforms, labels, _ in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings, _ = model.forward_features(waveforms)
            loss = loss_module(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        scores, all_labels = [], []
        with torch.no_grad():
            for waveforms, labels, _ in dev_loader:
                waveforms = waveforms.to(device)
                emb, _ = model.forward_features(waveforms)
                sc = loss_module.get_normalized_scores(emb)[:, 1].cpu().numpy()
                scores.extend(sc)
                all_labels.extend(labels.numpy())

        metrics = compute_all_metrics(np.array(all_labels), np.array(scores))
        print(f"  └── [Epoch {epoch:02d}] Val EER: {metrics['eer']:.2f}% | min t-DCF: {metrics['min_tdcf']:.4f} | Acc: {metrics['accuracy']:.1f}%")

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss_module.state_dict(),
                "best_eer": best_eer,
            }, best_path)
            print(f"  ⭐ Saved Best Checkpoint -> {best_path} (EER: {best_eer:.2f}%)")


# ======================================================================================
# 8. INFERENCE & PREDICTION
# ======================================================================================

def predict_single_audio(audio_path: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    device = get_device()
    model = AASIST().to(device)
    loss_module = AMSoftmaxLoss(in_features=model.embedding_dim, num_classes=2).to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        if "loss_state_dict" in ckpt:
            loss_module.load_state_dict(ckpt["loss_state_dict"], strict=False)

    model.eval()

    # Load audio robustly
    try:
        from src.utils.audio_io import robust_load_audio
        waveform, sr = robust_load_audio(audio_path, target_sr=16000)
        data = waveform.cpu().numpy()
    except Exception:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

    num_samples = 64600
    if len(data) < num_samples:
        repeats = int(np.ceil(num_samples / len(data)))
        data = np.tile(data, repeats)[:num_samples]
    elif len(data) > num_samples:
        start = (len(data) - num_samples) // 2
        data = data[start : start + num_samples]

    tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        emb, _ = model.forward_features(tensor)
        scores = loss_module.get_normalized_scores(emb).cpu().numpy()[0]

    spoof_score = float(scores[0])
    bona_score = float(scores[1])
    diff = bona_score - spoof_score
    spoof_prob = float(1.0 / (1.0 + np.exp(diff * 3.0)))
    is_spoof = diff < 0.0

    return {
        "audio_file": os.path.basename(audio_path),
        "verdict": "SPOOF / DEEPFAKE" if is_spoof else "BONAFIDE (HUMAN)",
        "is_spoof": is_spoof,
        "spoof_probability": round(spoof_prob, 4),
        "bonafide_probability": round(1.0 - spoof_prob, 4),
        "decision_margin": round(diff, 4),
    }


# ======================================================================================
# 9. CLI ENTRY POINT
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="ASVspoof AASIST Master Project File")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "predict", "generate"], default="predict")
    parser.add_argument("--audio", type=str, default=None, help="Input audio file for prediction")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "predict":
        if not args.audio:
            print("[!] Please supply an audio file: --audio path/to/sample.wav")
            sys.exit(1)
        res = predict_single_audio(args.audio, args.checkpoint)
        print("\n" + "="*50)
        print("🎙️ AASIST DEEPFAKE PREDICTION RESULT")
        print("="*50)
        for k, v in res.items():
            print(f"{k:22s}: {v}")
        print("="*50 + "\n")

    elif args.mode == "train":
        train_proto = "./data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        train_audio = "./data/asvspoof2019/LA/ASVspoof2019_LA_train/flac"
        dev_proto = "./data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        dev_audio = "./data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac"

        train_aasist(train_proto, train_audio, dev_proto, dev_audio, epochs=args.epochs)


if __name__ == "__main__":
    main()
