import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax (AM-Softmax) Loss for Anti-Spoofing and Deepfake Detection.
    
    Forces features of bona fide and spoofed utterances onto a hypersphere with an angular margin:
        L_i = -log( e^{s * (cos(theta_{y_i}) - m)} / (e^{s * (cos(theta_{y_i}) - m)} + sum_{j != y_i} e^{s * cos(theta_j)}) )
        
    Reference:
        Wang et al., "Additive Margin Softmax for Face Verification", IEEE SPL 2018.
    """

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

        # Weight matrix for hypersphere projection
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        if class_weights is not None:
            self.register_buffer("class_weights", torch.FloatTensor(class_weights))
        else:
            self.class_weights = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Latent feature vectors (Batch, in_features)
            labels: Ground truth class targets (Batch,)
        Returns:
            Scalar AM-Softmax loss
        """
        # Normalize weights and embeddings to unit hypersphere
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity: cos(theta) = x_norm @ w_norm^T
        costheta = F.linear(x_norm, w_norm) # (B, num_classes)

        # Apply angular margin to the ground truth class
        one_hot = torch.zeros_like(costheta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Adjusted cosine: cos(theta_y) - m
        costheta_m = costheta - self.m * one_hot

        # Scale by s
        scaled_logits = self.s * costheta_m

        # Cross entropy loss with optional class weighting
        loss = F.cross_entropy(scaled_logits, labels, weight=self.class_weights)
        return loss

    def get_normalized_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized cosine scores for inference / evaluation.
        Args:
            embeddings: (Batch, in_features)
        Returns:
            Cosine scores: (Batch, num_classes)
        """
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        return F.linear(x_norm, w_norm)
