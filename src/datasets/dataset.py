import os
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from .protocol_parser import AudioProtocolItem, parse_protocol_file
from .augmentations import RawBoost


class ASVSpoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 / 2021 combined voice anti-spoofing data.
    """

    def __init__(
        self,
        items: List[AudioProtocolItem],
        sample_rate: int = 16000,
        num_samples: int = 64600,
        is_train: bool = True,
        pad_mode: str = "repeat",
        slice_mode: str = "random",
        augmentor: Optional[RawBoost] = None,
    ):
        self.items = items
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.is_train = is_train
        self.pad_mode = pad_mode
        self.slice_mode = slice_mode if is_train else "center"
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio, resample to target sample_rate, convert to mono float32."""
        from src.utils.audio_io import robust_load_audio
        waveform, _ = robust_load_audio(filepath, target_sr=self.sample_rate)
        return waveform

        return waveform

    def _pad_or_slice(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform is exactly self.num_samples long."""
        length = waveform.size(0)

        if length == self.num_samples:
            return waveform

        if length < self.num_samples:
            # Padding
            if self.pad_mode == "repeat":
                repeat_times = int(np.ceil(self.num_samples / length))
                repeated = waveform.repeat(repeat_times)
                return repeated[: self.num_samples]
            else:
                pad_amt = self.num_samples - length
                return torch.nn.functional.pad(waveform, (0, pad_amt))
        else:
            # Slicing
            if self.slice_mode == "random" and self.is_train:
                max_start = length - self.num_samples
                start_idx = random.randint(0, max_start)
                return waveform[start_idx : start_idx + self.num_samples]
            else:
                # Center slice
                start_idx = (length - self.num_samples) // 2
                return waveform[start_idx : start_idx + self.num_samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]

        if not os.path.isfile(item.audio_path):
            # If file doesn't exist, generate synthetic zero/silence tensor to avoid crash
            waveform = torch.zeros(self.num_samples, dtype=torch.float32)
        else:
            try:
                waveform = self._load_audio(item.audio_path)
                waveform = self._pad_or_slice(waveform)
            except Exception:
                waveform = torch.zeros(self.num_samples, dtype=torch.float32)

        # Apply RawBoost if active
        if self.is_train and self.augmentor is not None:
            waveform = self.augmentor(waveform)

        return waveform, item.label, item.utt_id


def build_combined_dataset(
    protocol_configs: List[Dict[str, str]],
    sample_rate: int = 16000,
    num_samples: int = 64600,
    is_train: bool = True,
    pad_mode: str = "repeat",
    slice_mode: str = "random",
    augment: bool = False,
    rawboost_algo: int = 5,
) -> ASVSpoofDataset:
    """
    Build unified ASVSpoofDataset by merging multiple 2019 and 2021 protocols.
    """
    all_items: List[AudioProtocolItem] = []

    for cfg in protocol_configs:
        path = cfg["path"]
        audio_dir = cfg["audio_dir"]
        ver = cfg.get("dataset_version", "2019")
        if os.path.isfile(path) and os.path.isdir(audio_dir):
            items = parse_protocol_file(path, audio_dir, dataset_version=ver)
            all_items.extend(items)

    augmentor = RawBoost(algo=rawboost_algo) if (is_train and augment) else None

    return ASVSpoofDataset(
        items=all_items,
        sample_rate=sample_rate,
        num_samples=num_samples,
        is_train=is_train,
        pad_mode=pad_mode,
        slice_mode=slice_mode,
        augmentor=augmentor,
    )
