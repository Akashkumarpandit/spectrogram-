import os
import shutil
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def robust_load_audio(filepath: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Robust audio loader supporting WAV, FLAC, MP3, WebM (browser mic), OGG, M4A, etc.
    Fallbacks:
      1. soundfile (native C libsndfile)
      2. torchaudio.load (with torchcodec)
      3. librosa
      4. ffmpeg transcode
    """
    # 1. Try soundfile
    try:
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        waveform = torch.from_numpy(data).float()
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr
    except Exception:
        pass

    # 2. Try torchaudio (with torchcodec)
    try:
        waveform, sr = torchaudio.load(filepath)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr
    except Exception:
        pass

    # 3. Try librosa
    try:
        import librosa
        data, sr = librosa.load(filepath, sr=target_sr, mono=True)
        waveform = torch.from_numpy(data).float()
        return waveform, target_sr
    except Exception:
        pass

    # 4. Fallback to ffmpeg CLI transcode (converts any WebM/Opus/MP4/etc. to standard WAV)
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if os.path.isfile(ffmpeg_bin):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", filepath,
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "wav",
            tmp_wav_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.isfile(tmp_wav_path):
            try:
                data, sr = sf.read(tmp_wav_path, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                waveform = torch.from_numpy(data).float()
                os.remove(tmp_wav_path)
                return waveform, target_sr
            except Exception:
                if os.path.isfile(tmp_wav_path):
                    os.remove(tmp_wav_path)

    # 5. Fallback silent dummy tensor if completely unreadable
    return torch.zeros(target_sr * 4, dtype=torch.float32), target_sr
