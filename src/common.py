"""Utilidades compartidas del proyecto de clasificación de defectos.

Contiene el mismo código del notebook `entrenar_modelos.ipynb`
(arquitecturas, transformaciones, semilla y dispositivo) para que
`train.py`, `evaluate.py` y `predict.py` reproduzcan el experimento
sin depender de Jupyter.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

# Valores estándar de normalización utilizados por los modelos de ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Orden alfabético usado por torchvision.datasets.ImageFolder.
CLASS_NAMES = ["crack", "hole", "normal", "rust", "scratch"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAMES = [
    "cnn_baseline",
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_small",
]


def set_seed(seed: int = 42) -> None:
    """Fija la semilla de todos los generadores para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Estas opciones ayudan a mejorar la reproducibilidad en CUDA.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(mode: str = "auto") -> torch.device:
    """Selecciona el dispositivo de cómputo: auto/cuda/mps/cpu."""
    mode = mode.lower()

    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se forzó 'cuda' pero no hay GPU NVIDIA disponible.")
        return torch.device("cuda:0")

    if mode == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Se forzó 'mps' pero MPS no está disponible en este equipo.")
        return torch.device("mps")

    if mode == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Modo desconocido: {mode!r}. Usa 'auto', 'cuda', 'mps' o 'cpu'.")


def make_train_transform(img_size: int) -> transforms.Compose:
    """Transformaciones de entrenamiento (con data augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # El dataset es grayscale, pero los modelos preentrenados
        # esperan una entrada RGB de 3 canales.
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transform(img_size: int) -> transforms.Compose:
    """Transformaciones de validación/test (sin augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SimpleCNN(nn.Module):
    """CNN básica entrenada desde cero (baseline del proyecto)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Reduce cada mapa de características a un solo valor.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout ayuda a reducir sobreajuste.
            nn.Dropout(0.30),
            # Salida final: una neurona por clase.
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Construye un modelo del proyecto a partir de su nombre.

    pretrained=True descarga los pesos ImageNet (para entrenar).
    pretrained=False crea solo la arquitectura (para cargar un checkpoint,
    que ya contiene todos los pesos — así no se descarga nada).
    """
    name = name.lower()

    # CNN básica entrenada desde cero
    if name == "cnn_baseline":
        model = SimpleCNN(num_classes)

    # ResNet-18 preentrenada en ImageNet
    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # EfficientNet-B0 preentrenada en ImageNet
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # MobileNetV3-Small preentrenada en ImageNet
    elif name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Modelo no reconocido: {name}")

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Carga un checkpoint guardado por train.py y devuelve el modelo listo.

    Usa map_location=device, por lo que un modelo entrenado en GPU se
    puede cargar en CPU sin problema. El load es seguro (weights_only=True:
    el checkpoint solo contiene tensores y metadatos simples) y no descarga
    pesos ImageNet — el checkpoint ya contiene todos los pesos entrenados.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    """Evalúa un modelo sobre un DataLoader y devuelve métricas."""
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }
