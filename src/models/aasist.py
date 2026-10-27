import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .sincnet import SincConv
from .graph_modules import SpectroTemporalGraphFusion


class ResBlock(nn.Module):
    """
    2D Residual Convolutional block for spectro-temporal feature extraction.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: Tuple[int, int] = (2, 2)):
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
        res = x
        if self.res_conv is not None:
            res = self.res_bn(self.res_conv(res))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        out = self.pool(out)
        return out


class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.
    Processes raw audio waveforms end-to-end.
    """

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
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # 1. Front-end: Sinc-convolution on raw audio
        self.sinc_conv = SincConv(
            out_channels=sinc_filters,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            stride=1,
            padding=sinc_kernel // 2,
        )
        self.sinc_bn = nn.BatchNorm1d(sinc_filters)
        self.sinc_relu = nn.LeakyReLU(0.2)
        self.sinc_pool = nn.MaxPool1d(kernel_size=3, stride=3)

        # 2. Spectro-temporal 2D CNN encoder
        # Sinc output (B, 70, T) is reshaped into (B, 1, 70, T)
        self.res1 = ResBlock(1, 32, pool_size=(2, 3))
        self.res2 = ResBlock(32, 32, pool_size=(2, 3))
        self.res3 = ResBlock(32, 64, pool_size=(2, 3))
        self.res4 = ResBlock(64, 64, pool_size=(2, 3))

        # Channel projection to graph node dimension
        self.graph_proj = nn.Conv2d(64, 32, kernel_size=1)

        # 3. Heterogeneous Spectro-Temporal Graph Fusion
        self.graph_fusion = SpectroTemporalGraphFusion(in_dim=32, out_dim=32)
        # Graph fusion output: (B, 4 * out_dim * heads) = (B, 4 * 32 * 2) = (B, 256)
        fused_dim = 256

        # 4. Dense Projection & Classification Head
        self.fc_embed = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spectro-temporal features and normalized embedding.
        Args:
            x: Raw waveform tensor (Batch, Time) or (Batch, 1, Time)
        Returns:
            embedding: Normalized latent representation (Batch, embedding_dim)
            fused_rep: Raw graph representation (Batch, fused_dim)
        """
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        # Sinc convolution
        sinc_feats = self.sinc_pool(self.sinc_relu(self.sinc_bn(self.sinc_conv(x))))
        # Shape: (B, 70, T_reduced)

        # Reshape to (Batch, 1, Frequency_channels, Time)
        x_2d = sinc_feats.unsqueeze(1)

        # Pass through residual encoder
        x_2d = self.res1(x_2d)
        x_2d = self.res2(x_2d)
        x_2d = self.res3(x_2d)
        x_2d = self.res4(x_2d)

        # Project features: (B, C=32, F, T)
        x_graph = self.graph_proj(x_2d)
        B, C, F_sub, T_sub = x_graph.shape

        # Node representations:
        # Temporal nodes: aggregate along frequency subbands -> (B, T_sub, C)
        h_temporal = x_graph.mean(dim=2).permute(0, 2, 1)

        # Spectral nodes: aggregate along temporal frames -> (B, F_sub, C)
        h_spectral = x_graph.mean(dim=3).permute(0, 2, 1)

        # Heterogeneous Graph fusion
        fused_graph = self.graph_fusion(h_temporal, h_spectral)

        # Latent embedding
        embedding = self.fc_embed(fused_graph)

        return embedding, fused_graph

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Raw waveform tensor (Batch, Time)
        Returns:
            logits: (Batch, num_classes)
            embedding: (Batch, embedding_dim)
        """
        embedding, _ = self.forward_features(x)
        logits = self.classifier(embedding)
        return logits, embedding


class AASISTLight(AASIST):
    """
    Lightweight variant of AASIST (AASIST-L) with fewer channels for faster inference.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        sinc_filters: int = 70,
        sinc_kernel: int = 128,
        num_classes: int = 2,
        embedding_dim: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__(
            sample_rate=sample_rate,
            sinc_filters=sinc_filters,
            sinc_kernel=sinc_kernel,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        # Lighter residual blocks
        self.res1 = ResBlock(1, 16, pool_size=(2, 3))
        self.res2 = ResBlock(16, 16, pool_size=(2, 3))
        self.res3 = ResBlock(16, 32, pool_size=(2, 3))
        self.res4 = ResBlock(32, 32, pool_size=(2, 3))

        self.graph_proj = nn.Conv2d(32, 16, kernel_size=1)
        self.graph_fusion = SpectroTemporalGraphFusion(in_dim=16, out_dim=16)

        fused_dim = 128 # 4 * 16 * 2
        self.fc_embed = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
