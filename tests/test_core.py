import numpy as np
import torch
from torch import nn

from bicam import BiCAM, paper_overlay


class Attention(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    def forward(self, x):
        return x


class PatchMeta:
    num_patches = 4
    grid_size = (2, 2)


class TinyViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchMeta()
        self.blocks = nn.ModuleList([Attention() for _ in range(8)])
        self.head = nn.Linear(8, 3)

    def forward(self, tokens):
        for attention in self.blocks:
            tokens = tokens + attention(tokens)
        return self.head(tokens[:, 0])


def test_bicam_output_shape_and_signs():
    torch.manual_seed(7)
    model = TinyViT()
    explainer = BiCAM(model)
    result = explainer.generate(torch.randn(2, 5, 8))
    assert result.attribution.shape == (2, 1, 2, 2)
    assert torch.isfinite(result.attribution).all()
    assert torch.allclose(result.positive, torch.relu(result.attribution))
    assert torch.allclose(result.negative, torch.relu(-result.attribution))


def test_experiment_defaults():
    explainer = BiCAM(TinyViT())
    assert explainer.num_layers == 8
    assert explainer.temperature == 2.0


def test_paper_overlay_shape_and_range():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    attribution = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    overlay = paper_overlay(image, attribution)
    assert overlay.shape == image.shape
    assert 0.0 <= overlay.min() <= overlay.max() <= 1.0
