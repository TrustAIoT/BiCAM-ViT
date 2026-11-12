# Reproducibility guide

## Maintained architecture

- ViT-B/16 at 224 × 224

The primary ViT implementation follows the final August 2025 experiment
notebook. It aggregates the last eight transformer blocks, re-normalizes the
captured CLS attention logits with temperature `T = 2`, multiplies them by the
dot product between patch value vectors and the class gradient, and sums the
signed maps across heads and layers. The resulting 14 × 14 map is bilinearly
upsampled to the input resolution.

For the ImageNet visualizations, use a direct resize to 224 × 224, tensor
conversion, and channel-wise mean/std of `0.5`. Load the explicit checkpoint
`jx_vit_base_p16_224-80ecf9dd.pth`; `pretrained=True` alone may resolve to a
different timm checkpoint and does not reproduce the saved notebook output.

## Datasets

Datasets are intentionally not redistributed. Download ImageNet-1k, PASCAL VOC
2012, and COCO 2017 from their official sources and keep them outside this
repository. Dataset paths and generated artifacts are ignored by Git.

## Evaluation protocol

Localization maps are min-max scaled and thresholded at 0.5. Report pixel
accuracy, IoU, F1, precision, and recall. For multi-object datasets, evaluate
the positive channel against target-object annotations and the negative channel
against non-target annotations.

Faithfulness removes all 196 ViT-B/16 patches one by one. Compute both
least-important-first (LIF) and most-important-first (MIF) curves and report:

```text
Faithfulness = AUC_LIF - AUC_MIF
```

## Scope of this release

This repository intentionally contains only the maintained BiCAM method,
visualization helper, a minimal example, and lightweight tests. Historical
notebooks and architecture experiments remain in the private experiment
archive and are not required to use the released implementation.
