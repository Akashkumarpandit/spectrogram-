import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def to_hz(mel: float) -> float:
    """Convert Mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class SincConv(nn.Module):
    """
    Sinc-based convolution layer for raw audio representation.
    Parametric band-pass filter based on sinc functions:
        h[n] = 2 * f2 * sinc(2 * pi * f2 * n) - 2 * f1 * sinc(2 * pi * f1 * n)
    modulated by a Hamming window.
    
    Reference:
        Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet", SLT 2018.
    """

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 128,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 0.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports in_channels=1, got {in_channels}")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filter cutoffs evenly distributed in Mel scale
        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(to_mel(low_hz), to_mel(high_hz), self.out_channels + 1)
        hz = to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # Time vector
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: Tensor of shape (Batch, 1, Time) or (Batch, Time)
        Returns:
            Filtered feature maps: Tensor of shape (Batch, out_channels, Time_out)
        """
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
