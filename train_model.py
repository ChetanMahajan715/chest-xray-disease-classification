import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_utils import build_model, save_class_names


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(data_dir: str, image_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader, list]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.classes


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / max(total, 1),
        "acc": running_correct / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--freeze_features", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = make_dataloaders(args.data_dir, args.image_size, args.batch_size)
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=len(class_names)).to(device)
    if args.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-5,
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    model_path = os.path.join(args.model_dir, "densenet121_chestxray.pt")
    classes_path = os.path.join(args.model_dir, "class_names.json")
    history_path = os.path.join(args.model_dir, "train_history.json")

    save_class_names(classes_path, class_names)

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                },
                model_path,
            )
            print(f"Saved best model to: {model_path}")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Class names saved to: {classes_path}")


if __name__ == "__main__":
    main()
