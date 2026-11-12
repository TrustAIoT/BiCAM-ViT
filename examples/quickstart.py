"""Reproduce a paper-style BiCAM overlay with an ImageNet-pretrained ViT."""

from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

from bicam import BiCAM, paper_overlay


device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch16_224", pretrained=False).to(device)
weights = torch.hub.load_state_dict_from_url(
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
    map_location=device,
)
model.load_state_dict(weights, strict=True)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
explainer = BiCAM(model)

image_path = Path(__file__).parents[1] / "assets" / "example_barn.png"
image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)
result = explainer.generate(tensor, target=425)  # ImageNet class 425: barn

display = (tensor[0].detach().cpu() * 0.5 + 0.5).clamp(0, 1)
display = display.permute(1, 2, 0).numpy()
overlay = paper_overlay(display, result.attribution[0, 0].cpu().numpy())
output_path = Path("bicam_barn_overlay.png")
Image.fromarray(np.uint8(overlay * 255)).save(output_path)

print("prediction:", result.prediction.item())
print("signed map:", tuple(result.attribution.shape))
print("saved:", output_path.resolve())
explainer.restore_model()
