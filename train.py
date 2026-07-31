"""
train.py — Train an image tagging / classification model.

Approach
--------
Instead of training a CNN from scratch (which needs huge datasets and
compute), this uses TRANSFER LEARNING: it starts from a MobileNetV2
already trained on ImageNet (1.4M images, 1000 classes) and re-trains
only the final layer(s) to recognize YOUR classes (e.g. cat, dog, car).

This is the standard, practical approach used in real production
systems when you don't have millions of labeled images.

Expected folder structure
--------------------------
data/
    train/
        cat/   img1.jpg  img2.jpg ...
        dog/   img1.jpg  img2.jpg ...
        car/   img1.jpg  img2.jpg ...
    val/
        cat/   ...
        dog/   ...
        car/   ...

Usage
-----
    python train.py --data_dir data --epochs 10 --output model.pt

Output
------
Saves a single file (model.pt) containing the trained weights AND the
list of class names, so predict.py can load everything it needs from
one file.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(num_classes: int) -> nn.Module:
    """Load a pretrained MobileNetV2 and swap the final layer for our classes."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the pretrained "feature extractor" layers — we only train
    # the final classifier layer. This is fast and needs little data.
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_dataloaders(data_dir: str, batch_size: int = 32):
    # Standard ImageNet normalization stats — required since we're
    # reusing ImageNet-pretrained weights.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform) if val_dir.exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        if val_ds is not None else None
    )

    return train_loader, val_loader, train_ds.classes


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Found classes: {classes}")

    model = build_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    # Only the unfrozen params (final layer) need an optimizer.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        msg = f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}"

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            msg += f" - val_acc: {val_acc:.4f}"
            if val_acc > best_acc:
                best_acc = val_acc

        msg += f" - time: {time.time() - start:.1f}s"
        print(msg)

    # Save weights + class names together so predict.py is self-contained.
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
    }, args.output)
    print(f"\nSaved trained model to {args.output}")

    # Also save a human-readable class list.
    with open(Path(args.output).with_suffix(".classes.json"), "w") as f:
        json.dump(classes, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image tagging model.")
    parser.add_argument("--data_dir", type=str, default="data",
                         help="Path to data folder containing train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="model.pt",
                         help="Where to save the trained model")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
