import json
import os
from typing import List

import torch
from torchvision.models import DenseNet121_Weights, densenet121


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model


def save_class_names(path: str, class_names: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
