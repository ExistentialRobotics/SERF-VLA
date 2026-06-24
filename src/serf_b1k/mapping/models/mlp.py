import torch
import torch.nn as nn

__all__ = ["MLP"]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def init_weights_kaiming(m):
    """Kaiming Initialization - Better suited for ReLU/GELU."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ResidualBlock(nn.Module):
    """Pre-LN residual block with expansion-contraction pattern."""
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# --------------------------------------------------------------------------- #
#  MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """
    Generic MLP with residual blocks.

    Used as both encoder (DINO → latent) and decoder (latent → DINO).

    Architecture:
        input → Linear → LayerNorm → GELU → [ResBlock × N] → LayerNorm → Linear → output
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 384,
        num_res_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # Final normalization before output projection
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection (no activation)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.apply(init_weights_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) input features

        Returns:
            (B, output_dim) output features
        """
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x
