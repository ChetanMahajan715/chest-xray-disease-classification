import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, overlay_heatmap
from model_utils import build_model


def load_model(model_path: str, device: torch.device):
    """Load trained model and class names."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


def predict_image(model: torch.nn.Module, image_path: str, class_names: list, image_size: int, device: torch.device):
    """Predict class for a single image."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    return pred_label, confidence, input_tensor, pred_idx, probs


def main():
    parser = argparse.ArgumentParser(description="Chest X-Ray Disease Classification - Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image")
    parser.add_argument("--model_dir", default="models", help="Directory containing model checkpoint")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save output images")
    parser.add_argument("--show_heatmap", action="store_true", help="Generate and save Grad-CAM heatmap")
    parser.add_argument("--top_k", type=int, default=3, help="Show top K predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = os.path.join(args.model_dir, "densenet121_chestxray.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run training first: python train_model.py")
        return

    model, class_names, image_size = load_model(model_path, device)
    print(f"Loaded model with classes: {class_names}")

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    pred_label, confidence, input_tensor, pred_idx, probs = predict_image(
        model, args.image, class_names, image_size, device
    )

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Image: {args.image}")
    print(f"Predicted Class: {pred_label}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50)

    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 40)
    top_k = min(args.top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, top_k)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        label = class_names[int(idx)]
        print(f"  {i + 1}. {label}: {float(prob) * 100:.2f}%")

    if args.show_heatmap:
        os.makedirs(args.output_dir, exist_ok=True)
        gradcam = GradCAM(model)
        heatmap, _ = gradcam.generate(input_tensor, class_idx=pred_idx)

        image_np = np.array(Image.open(args.image).convert("RGB"))
        overlay = overlay_heatmap(image_np, heatmap)

        output_filename = f"gradcam_{os.path.basename(args.image)}"
        output_path = os.path.join(args.output_dir, output_filename)

        import cv2

        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\nGrad-CAM heatmap saved to: {output_path}")

    print("\n")


if __name__ == "__main__":
    main()
