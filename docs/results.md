# Reported results

These numbers are transcribed from the camera-ready manuscript. See the paper
for the full protocol, baselines, uncertainty analysis, and discussion.

## Localization

| Dataset / method | Pixel accuracy | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| ImageNet — BiCAM | 0.6253 | **0.5419** | **0.6624** | 0.5901 | **0.9288** |
| VOC 2012 — BiCAM positive | 0.8559 | **0.3700** | **0.5104** | **0.6095** | **0.5863** |
| VOC 2012 — BiCAM negative | 0.7705 | 0.2588 | 0.3786 | 0.3642 | 0.5779 |
| COCO 2017 — BiCAM positive | 0.8707 | **0.2971** | **0.4191** | **0.5535** | 0.5154 |
| COCO 2017 — BiCAM negative | 0.8487 | 0.1141 | 0.1724 | 0.2711 | 0.2074 |

## Faithfulness

| Dataset | LIF ↑ | MIF ↓ | Faithfulness ↑ |
|---|---:|---:|---:|
| ImageNet | **0.5478** | 0.1654 | **0.3824** |
| VOC 2-class | **0.9313** | **0.4686** | **0.4626** |
| COCO 2-class | **0.9407** | **0.4997** | **0.4410** |

## Runtime

On ViT-B/16 with an RTX 4090, BiCAM reports **16.0 ms/image** and
**0.24 MB/image**, with no training overhead.
