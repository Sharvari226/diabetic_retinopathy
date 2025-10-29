# src/visualize.py
import torch
import yaml
import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import DRMultiTaskModel
from src.utils import autocrop_background
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def load_model_for_cam(ckpt_path, device='cpu'):
    model = DRMultiTaskModel(backbone_name='efficientnet_b0', pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in state:
        sd = state['state_dict']
        new_sd = {}
        for k,v in sd.items():
            if k.startswith('model.'):
                new_sd[k.replace('model.','')] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def prepare_image(img_path, image_size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = autocrop_background(img)
    orig = img.copy().astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    input_tensor = transform(orig).unsqueeze(0)
    return orig, input_tensor

def run_gradcam(ckpt, img_path, output_path, layer_name=None, device='cuda'):
    model = load_model_for_cam(ckpt, device=device)
    orig, input_tensor = prepare_image(img_path)
    input_tensor = input_tensor.to(device)
    # pick final conv layer
    if layer_name is None:
        # find last conv feature map in backbone
        target_layer = model.model.backbone.conv_head if hasattr(model.model.backbone, 'conv_head') else None
        # fallback: use backbone feature extractor output layer
        # For timm efficientnet_b0, features returns last conv in .blocks[-1]
        target_layer = model.model.backbone.blocks[-1] if hasattr(model.model.backbone, 'blocks') else target_layer
    else:
        target_layer = layer_name
    cam = GradCAM(model=model.model.backbone, target_layers=[target_layer], use_cuda=(device=='cuda'))
    # target is positive referable class (sigmoid); grad-cam needs class index; here we use logits from fc_dr
    targets = [ClassifierOutputTarget(1)]  # might be ignored; alternative approach: pass in logit
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(8,8))
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print("Saved CAM to", output_path)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    ckpt = cfg['logging']['model_ckpt']
    # Call function: run_gradcam(ckpt, "path/to/image.jpg", "out.png")
