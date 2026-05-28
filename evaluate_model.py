import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_utils import build_model


def make_test_loader(data_dir: str, image_size: int, batch_size: int) -> DataLoader:
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(args.model_dir, "densenet121_chestxray.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = make_test_loader(args.data_dir, args.image_size, args.batch_size)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification report:")
    print(report)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
