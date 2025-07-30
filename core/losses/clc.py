import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentAwareSemanticModulation(nn.Module):
    def __init__(self, d):
        """
        Segment-aware semantic modulation module.
        d: feature dimensionality
        """
        super(SegmentAwareSemanticModulation, self).__init__()
        self.d = d

        # Gating network: W_g ∈ R^{d x 2d}, b_g ∈ R^d
        self.gate_fc = nn.Linear(2 * d, d)
        self.sigmoid = nn.Sigmoid()

        # Fusion network: φ_f: R^{T_i x 2d} → R^{T_i x d}
        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.LayerNorm(d),
            nn.Linear(d, d)
        )

    def forward(self, X_i, y):
        """
        X_i: Tensor of shape [T_i, d] - motion segment token
        y: Tensor of shape [1, d]     - global text token
        Returns: segment-aware modulated token of shape [T_i, d]
        """
        T_i, d = X_i.size()

        # Temporal average pooling → [1, d]
        X_avg = X_i.mean(dim=0, keepdim=True)

        # Concatenate [AvgPool(X_i), y] → [1, 2d]
        concat_input = torch.cat([X_avg, y], dim=-1)

        # Gating vector γ_i ∈ [1, d]
        gamma_i = self.sigmoid(self.gate_fc(concat_input))

        # Channel-wise gating → [1, d]
        y_hat_i = gamma_i * y

        # Repeat modulated y across T_i steps → [T_i, d]
        y_hat_repeated = y_hat_i.repeat(T_i, 1)

        # Concatenate segment + text → [T_i, 2d]
        fusion_input = torch.cat([X_i, y_hat_repeated], dim=-1)

        # Segment-aware fusion → [T_i, d]
        X_i_hat = self.fusion_fc(fusion_input)
        return X_i_hat


class CrossLevelConsistencyLoss(nn.Module):
    def __init__(self, d):
        """
        Cross-Level Consistency Loss.
        d: latent dimensionality
        """
        super(CrossLevelConsistencyLoss, self).__init__()
        self.projection = nn.Linear(d, d)

    def forward(self, fs_list):
        """
        fs_list: list of latent codes FS_1, FS_2, FS_3, FS_4
                 each of shape [B, d]
        Computes: ∑_{i=1}^{3} || FS_i - ψ(FS_{i+1}) ||_2^2
        """
        loss = 0.0
        for i in range(len(fs_list) - 1):
            fs_i = fs_list[i]
            fs_ip1 = self.projection(fs_list[i + 1])
            loss += F.mse_loss(fs_i, fs_ip1)
        return loss

    def __repr__(self):
        return f"CrossLevelConsistencyLoss(d={self.projection.in_features})"


# Example usage
if __name__ == "__main__":
    d = 256       # feature dimension
    T_i = 20      # length of segment
    B = 32        # batch size

    # Segment-aware semantic modulation test
    X_i = torch.randn(T_i, d)         # motion segment
    y = torch.randn(1, d)             # global text embedding

    segment_mod = SegmentAwareSemanticModulation(d)
    X_i_hat = segment_mod(X_i, y)
    print("Segment-aware modulated output:", X_i_hat.shape)  # [T_i, d]

    # Cross-level consistency loss test
    fs_list = [torch.randn(B, d) for _ in range(4)]
    clc_loss_fn = CrossLevelConsistencyLoss(d)
    loss = clc_loss_fn(fs_list)
    print("Cross-level consistency loss:", loss.item())
