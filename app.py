import os
from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, overlay_heatmap
from model_utils import build_model


MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "densenet121_chestxray.pt")
DEFAULT_IMAGE_SIZE = 224


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@st.cache_resource
def load_model() -> Tuple[torch.nn.Module, list, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


def predict_image(model: torch.nn.Module, image: Image.Image, image_size: int):
    device = next(model.parameters()).device
    transform = get_transform(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return tensor, probs.cpu().numpy()


def main() -> None:
    st.title("Chest X-Ray Disease Classification")
    st.caption("Educational decision-support tool. Not for clinical diagnosis.")

    if not os.path.exists(CHECKPOINT_PATH):
        st.error("Model file not found. Run training first: python train_model.py")
        return

    model, class_names, image_size = load_model()

    uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running prediction..."):
            input_tensor, probs = predict_image(model, image, image_size)
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100.0

        st.subheader("Prediction")
        st.write(f"{pred_label} ({confidence:.2f}%)")

        try:
            gradcam = GradCAM(model)
            heatmap, _ = gradcam.generate(input_tensor, class_idx=pred_idx)
            overlay = overlay_heatmap(image_np, heatmap)

            st.subheader("Grad-CAM")
            st.image(overlay, caption="Highlighted regions used by the model", use_container_width=True)
        except Exception as exc:
            st.warning(f"Prediction worked, but Grad-CAM could not be generated: {exc}")
    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")


if __name__ == "__main__":
    main()
