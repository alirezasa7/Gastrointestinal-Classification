
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

from MSR import MSR  


class GradCAM:
    def __init__(self, model, target_module=None, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_module = target_module
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        if target_module and target_layer is None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_module.register_forward_hook(forward_hook)
        self.target_module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)  # Run forward pass

        if self.target_layer is not None:
            self.activations = self.model.activations[self.target_layer]
            self.activations.retain_grad()
        elif self.activations is None:
            raise ValueError("No activations captured")

        if target_class is None:
            target_class = output.argmax().item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        if self.target_layer is not None:
            self.gradients = self.activations.grad

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()  # ← FIXED HERE
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam


def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img = Image.open(img_path).convert('RGB')
    return preprocess(img).unsqueeze(0), img


def overlay_cam_on_image(img, cam, alpha=0.5):
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = img.resize((cam.shape[1], cam.shape[0]))
    img_np = np.array(img)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/kaggle/working/best_model (1).pth'
model = MSR(num_classes=22)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


gradcams = {
    'x_low': GradCAM(model, target_layer='x_low'),
    'x_medium': GradCAM(model, target_layer='x_medium'),
    'x_high': GradCAM(model, target_layer='x_high'),
    'Cross-Attention': GradCAM(model, model.cross_attn),
    'CBAM': GradCAM(model, model.cbam),
}


data_root = '/kaggle/working/gastrovision_split/train'
class_folders = sorted(os.listdir(data_root))

for class_name in class_folders:
    class_path = os.path.join(data_root, class_name)
    img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)

    input_tensor, original_img = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Grad-CAM overlays for Class: {class_name}", fontsize=18)

    # Show original image
    plt.subplot(1, len(gradcams) + 1, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis('off')

    # Grad-CAM overlays for each module
    for i, (name, gradcam) in enumerate(gradcams.items(), start=2):
        cam = gradcam.generate_cam(input_tensor)
        overlay = overlay_cam_on_image(original_img, cam)

        plt.subplot(1, len(gradcams) + 1, i)
        plt.imshow(overlay)
        plt.title(f"Grad-CAM: {name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
