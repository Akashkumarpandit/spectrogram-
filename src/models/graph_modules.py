import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention Layer with residual connection.
    Computes dynamic attention coefficients between graph nodes.
    """

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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature tensor (Batch, NumNodes, in_features)
        Returns:
            Updated node feature tensor (Batch, NumNodes, out_features * num_heads)
        """
        B, N, _ = h.shape

        # Linear projection: (B, N, num_heads, out_features)
        h_prime = self.W(h).view(B, N, self.num_heads, self.out_features)

        # Attention scores: a_src^T * h_i + a_dst^T * h_j
        attn_src = torch.einsum("bnhd,hdo->bnh", h_prime, self.a_src) # (B, N, num_heads)
        attn_dst = torch.einsum("bnhd,hdo->bnh", h_prime, self.a_dst) # (B, N, num_heads)

        # Pairwise attention: (B, num_heads, N, N)
        attn_src = attn_src.permute(0, 2, 1).unsqueeze(-1) # (B, H, N, 1)
        attn_dst = attn_dst.permute(0, 2, 1).unsqueeze(-2) # (B, H, 1, N)
        attn = self.leaky_relu(attn_src + attn_dst) / self.temperature

        attn_weights = F.softmax(attn, dim=-1)

        # Message aggregation: (B, H, N, D)
        h_perm = h_prime.permute(0, 2, 1, 3) # (B, H, N, D)
        out = torch.matmul(attn_weights, h_perm) # (B, H, N, D)

        # Reshape: (B, N, H * D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        # Add residual connection
        return F.relu(out + self.res_proj(h))


class GraphPooling(nn.Module):
    """
    Learnable Graph Pooling layer that scores nodes and selects top-k informative nodes.
    """

    def __init__(self, in_features: int, pool_ratio: float = 0.5):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.score_proj = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (Batch, NumNodes, in_features)
        Returns:
            Pooled node tensor: (Batch, k, in_features)
        """
        B, N, D = h.shape
        k = max(1, int(N * self.pool_ratio))

        scores = torch.sigmoid(self.score_proj(h)).squeeze(-1) # (B, N)
        _, topk_indices = torch.topk(scores, k=k, dim=-1) # (B, k)

        # Gather top-k nodes
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, k)
        pooled = h[batch_idx, topk_indices] # (B, k, D)

        # Weight nodes by their normalized gating scores
        gates = torch.gather(scores, dim=-1, index=topk_indices).unsqueeze(-1) # (B, k, 1)
        return pooled * gates


class SpectroTemporalGraphFusion(nn.Module):
    """
    Heterogeneous Graph Module for AASIST.
    Takes spectral and temporal node representations, applies cross-attention
    and merges them into a global spectro-temporal graph representation.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.temporal_gat = GraphAttentionLayer(in_dim, out_dim, num_heads=2)
        self.spectral_gat = GraphAttentionLayer(in_dim, out_dim, num_heads=2)
        
        total_dim = out_dim * 2 # 2 heads
        self.temporal_pool = GraphPooling(total_dim, pool_ratio=0.5)
        self.spectral_pool = GraphPooling(total_dim, pool_ratio=0.5)

        # Cross-domain bidirectional fusion
        self.cross_attn_s2t = nn.MultiheadAttention(embed_dim=total_dim, num_heads=2, batch_first=True)
        self.cross_attn_t2s = nn.MultiheadAttention(embed_dim=total_dim, num_heads=2, batch_first=True)

        self.norm_t = nn.LayerNorm(total_dim)
        self.norm_s = nn.LayerNorm(total_dim)

    def forward(self, h_temporal: torch.Tensor, h_spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_temporal: (Batch, NumFrames, InDim)
            h_spectral: (Batch, NumBands, InDim)
        Returns:
            Fused global representation: (Batch, TotalDim * 2)
        """
        # Graph attention updates within each domain
        h_t = self.temporal_gat(h_temporal)
        h_s = self.spectral_gat(h_spectral)

        # Pool to salient nodes
        h_t = self.temporal_pool(h_t)
        h_s = self.spectral_pool(h_s)

        # Cross-attention across time and frequency domains
        t_fused, _ = self.cross_attn_s2t(query=h_t, key=h_s, value=h_s)
        s_fused, _ = self.cross_attn_t2s(query=h_s, key=h_t, value=h_t)

        h_t = self.norm_t(h_t + t_fused)
        h_s = self.norm_s(h_s + s_fused)

        # Readout: Max & Mean pooling across graph nodes
        t_rep = torch.cat([h_t.mean(dim=1), h_t.max(dim=1)[0]], dim=-1) # (B, 2 * TotalDim)
        s_rep = torch.cat([h_s.mean(dim=1), h_s.max(dim=1)[0]], dim=-1) # (B, 2 * TotalDim)

        # Concatenate spectro-temporal representations
        return torch.cat([t_rep, s_rep], dim=-1)
