from typing import Optional, Tuple

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        if target_layer is None:
            # DenseNet121 last convolution in features
            target_layer = model.features.denseblock4.denselayer16.conv2
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        heatmap = cam.cpu().numpy()
        return heatmap, class_idx


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, image_rgb, 1 - alpha, 0)
    return overlay
