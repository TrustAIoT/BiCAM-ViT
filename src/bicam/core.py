"""BiCAM implementation used by the final NeurIPS experiment notebooks.

The released method aggregates the last eight ViT attention blocks. For every
block it combines temperature-normalized CLS attention with the dot product
between patch value vectors and the class-specific gradient of the per-head
attention output. Summation across heads and layers preserves both supportive
and suppressive evidence.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class BiCAMOutput:
    """Outputs returned by :meth:`BiCAM.generate`."""

    prediction: Tensor
    target: Tensor
    attribution: Tensor
    positive: Tensor
    negative: Tensor


def _instrument_attention(module: nn.Module) -> None:
    """Install the observable attention path used by the experiment notebook."""

    if hasattr(module, "_bicam_original_forward"):
        return

    module._bicam_original_forward = module.forward  # type: ignore[attr-defined]

    def forward(self: nn.Module, x: Tensor) -> Tensor:
        # Intentional: this detach is present in the August 2025 experiment
        # hook and is required to reproduce the saved notebook results.
        x = x.detach()
        batch, tokens, channels = x.shape
        heads = self.num_heads  # type: ignore[attr-defined]
        qkv = self.qkv(x).reshape(  # type: ignore[attr-defined]
            batch, tokens, 3, heads, channels // heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        if hasattr(self, "q_norm"):
            query = self.q_norm(query)  # type: ignore[attr-defined]
        if hasattr(self, "k_norm"):
            key = self.k_norm(key)  # type: ignore[attr-defined]

        attention_logits = (
            query @ key.transpose(-2, -1)
        ) * self.scale  # type: ignore[attr-defined]
        attention = attention_logits.softmax(dim=-1)
        attention = self.attn_drop(attention)  # type: ignore[attr-defined]

        head_output = attention @ value
        if head_output.requires_grad:
            head_output.retain_grad()

        self._bicam_attention_logits = attention_logits  # type: ignore[attr-defined]
        self._bicam_values = value  # type: ignore[attr-defined]
        self._bicam_head_output = head_output  # type: ignore[attr-defined]

        output = head_output.transpose(1, 2).reshape(batch, tokens, channels)
        if hasattr(self, "norm"):
            output = self.norm(output)  # type: ignore[attr-defined]
        output = self.proj(output)  # type: ignore[attr-defined]
        output = self.proj_drop(output)  # type: ignore[attr-defined]
        return output

    module.forward = types.MethodType(forward, module)


def _attention_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    for module in model.modules():
        # Standard timm ViT blocks expose qkv, num_heads, and proj.
        if (
            module.__class__.__name__ == "Attention"
            and hasattr(module, "qkv")
            and hasattr(module, "num_heads")
            and hasattr(module, "proj")
        ):
            modules.append(module)
    return modules


class BiCAM:
    """Generate signed class-specific maps for timm ViT classifiers.

    Args:
        model: A timm-style Vision Transformer classifier.
        num_layers: Number of final transformer blocks to aggregate. The paper
            notebook uses eight.
        temperature: Temperature applied before re-normalizing the captured
            CLS attention logits. The paper notebook uses 2.0.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int = 8,
        temperature: float = 2.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.num_layers = num_layers
        self.temperature = temperature
        self.attention_modules = _attention_modules(model)
        if len(self.attention_modules) < num_layers:
            raise ValueError(
                f"BiCAM needs {num_layers} compatible Attention blocks, "
                f"but found {len(self.attention_modules)}"
            )
        for module in self.attention_modules:
            _instrument_attention(module)

    def _clear_capture(self) -> None:
        names = (
            "_bicam_attention_logits",
            "_bicam_values",
            "_bicam_head_output",
        )
        for module in self.attention_modules:
            for name in names:
                if hasattr(module, name):
                    delattr(module, name)

    def _grid_shape(self, token_count: int) -> tuple[int, int, int]:
        patch_embed = getattr(self.model, "patch_embed", None)
        num_patches = getattr(patch_embed, "num_patches", None)
        grid_size = getattr(patch_embed, "grid_size", None)

        if num_patches is None:
            num_patches = token_count - 1
        special_tokens = token_count - int(num_patches)
        if special_tokens < 0:
            raise RuntimeError("Captured fewer tokens than the model's patch count")

        if grid_size is not None:
            if isinstance(grid_size, Iterable):
                height, width = tuple(int(value) for value in grid_size)
            else:
                height = width = int(grid_size)
        else:
            height = int(int(num_patches) ** 0.5)
            width = int(num_patches) // height

        if height * width != int(num_patches):
            raise RuntimeError("Patch tokens cannot be reshaped into a spatial grid")
        return special_tokens, height, width

    def generate(
        self,
        images: Tensor,
        target: int | Tensor | None = None,
    ) -> BiCAMOutput:
        """Run one forward/backward pass and return the paper-faithful signed map.

        Image inputs return a map upsampled to the input resolution. Token-only
        inputs, useful for testing, return the native patch-grid resolution.
        """

        self._clear_capture()
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError("Expected classifier output with shape [batch, classes]")

        prediction = logits.argmax(dim=1)
        if target is None:
            targets = prediction
        elif isinstance(target, int):
            targets = torch.full_like(prediction, target)
        else:
            targets = target.to(device=logits.device, dtype=torch.long).reshape(-1)
            if targets.numel() != logits.shape[0]:
                raise ValueError("target must contain one class index per image")

        score = logits.gather(1, targets[:, None]).sum()
        score.backward()

        layer_maps: list[Tensor] = []
        selected = reversed(self.attention_modules[-self.num_layers :])
        for module in selected:
            attention_logits = getattr(module, "_bicam_attention_logits", None)
            values = getattr(module, "_bicam_values", None)
            head_output = getattr(module, "_bicam_head_output", None)
            if (
                attention_logits is None
                or values is None
                or head_output is None
                or head_output.grad is None
            ):
                raise RuntimeError("BiCAM could not capture an attention block")

            attention = torch.softmax(
                attention_logits[:, :, 0:1, :] / self.temperature,
                dim=-1,
            )
            cls_gradient = head_output.grad[:, :, 0:1, :]
            value_gradient = values @ cls_gradient.transpose(-2, -1)
            signed = value_gradient.transpose(-2, -1) * attention
            layer_maps.append(signed.sum(dim=1).squeeze(1))

        token_map = torch.stack(layer_maps, dim=0).sum(dim=0)
        token_count = token_map.shape[-1]
        special_tokens, height, width = self._grid_shape(token_count)
        attribution = token_map[..., special_tokens:].reshape(-1, 1, height, width)

        if images.ndim == 4:
            attribution = F.interpolate(
                attribution,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return BiCAMOutput(
            prediction=prediction.detach(),
            target=targets.detach(),
            attribution=attribution.detach(),
            positive=torch.relu(attribution).detach(),
            negative=torch.relu(-attribution).detach(),
        )

    def restore_model(self) -> None:
        """Restore the original forward method of every instrumented block."""

        for module in self.attention_modules:
            original = getattr(module, "_bicam_original_forward", None)
            if original is not None:
                module.forward = original
                delattr(module, "_bicam_original_forward")
        self._clear_capture()
