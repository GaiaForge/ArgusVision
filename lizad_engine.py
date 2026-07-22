"""
LiZAD inference engine, kept separate from app.py's camera/UI code.

Based on reading LiZAD's actual source (github.com/intelligolabs/LiZAD):
- model/model.py: ZSADModel(args) with vision_layers, text_dim, vision_dim, out_dim, img_size
- backbones/encoders.py: ImageEncoder(model_id, layers, device), TextEncoder(model_id, device)
- pipeline/test.py: checkpoint loading + text_embeddings_dict construction pattern
- utils/transformations.py: resize to img_size (bicubic) + ImageNet normalization
- datasets/constants.py: normal/abnormal prompt templates

NOTE: this hasn't been run against real hardware yet - the overall shape of these
calls is verified against LiZAD's source, but expect to debug on first real run
on the Jetson, same as every other piece of this project so far.
"""

import types
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from backbones.encoders import ImageEncoder, TextEncoder
from model.model import ZSADModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORMAL_TEMPLATES = ["{}", "flawless {}", "perfect {}", "{} without flaw", "{} without defect"]
ABNORMAL_TEMPLATES = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]

TIPS_ARGS = types.SimpleNamespace(
    vision_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    text_model_id="MobileCLIP2-S0",
    vision_layers=[3, 5, 7, 11],
    text_dim=512,
    vision_dim=384,
)
GENERIC_ARGS = types.SimpleNamespace(img_size=518, out_dim=256)


class LiZADEngine:
    def __init__(self, checkpoint_path, class_name="pcb", device="cuda:0"):
        self.device = device
        self.class_name = class_name

        self.image_encoder = ImageEncoder(
            model_id=TIPS_ARGS.vision_model_id,
            layers=TIPS_ARGS.vision_layers,
            device=device,
        )
        self.text_encoder = TextEncoder(model_id=TIPS_ARGS.text_model_id, device=device)

        model_args = types.SimpleNamespace(
            vision_layers=TIPS_ARGS.vision_layers,
            text_dim=TIPS_ARGS.text_dim,
            vision_dim=TIPS_ARGS.vision_dim,
            out_dim=GENERIC_ARGS.out_dim,
            img_size=GENERIC_ARGS.img_size,
        )
        self.model = ZSADModel(model_args).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((GENERIC_ARGS.img_size, GENERIC_ARGS.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.text_embeddings_dict = self._build_text_embeddings(class_name)

    def _build_text_embeddings(self, class_name):
        normal_prompts = [t.format(class_name) for t in NORMAL_TEMPLATES]
        abnormal_prompts = [t.format(class_name) for t in ABNORMAL_TEMPLATES]
        with torch.no_grad():
            normal_batch = self.text_encoder(normal_prompts).mean(dim=0, keepdim=True)
            abnormal_batch = self.text_encoder(abnormal_prompts).mean(dim=0, keepdim=True)
        return {"normal": normal_batch, "abnormal": abnormal_batch}

    @torch.no_grad()
    def run(self, frame_rgb):
        """frame_rgb: HxWx3 numpy array (RGB, uint8). Returns (anomaly_map_2d, score)."""
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        cls, patches = self.image_encoder(img_tensor)
        anomaly_map, _ = self.model(self.text_embeddings_dict, [cls, patches])

        # channel 1 = "abnormal" class after softmax
        pixel_map = anomaly_map[0, 1].detach().cpu().numpy()
        score = float(pixel_map.max())
        return pixel_map, score


def overlay_heatmap(frame_rgb, anomaly_map, alpha=0.45):
    """Resize anomaly_map to frame size, colorize, and alpha-blend onto frame_rgb."""
    h, w = frame_rgb.shape[:2]
    resized = cv2.resize(anomaly_map, (w, h))
    normalized = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended
