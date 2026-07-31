"""
predict.py — Tag new images using a model trained with train.py.

Usage
-----
    # Tag a single image
    python predict.py --model model.pt --image photo.jpg

    # Tag every image in a folder
    python predict.py --model model.pt --image_dir my_photos/

    # Also works with the built-in ImageNet model if you never trained
    # your own (1000 general categories, no training needed):
    python predict.py --pretrained --image photo.jpg
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_custom_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, classes


def load_pretrained_model(device):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights).to(device).eval()
    classes = weights.meta["categories"]
    return model, classes


def tag_image(image_path, model, classes, transform, device, top_k=3):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = probs.topk(min(top_k, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top_probs, top_idxs)]


def collect_images(args):
    if args.image:
        return [Path(args.image)]
    if args.image_dir:
        folder = Path(args.image_dir)
        return sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise ValueError("Provide either --image or --image_dir")


def main():
    parser = argparse.ArgumentParser(description="Tag images with a trained (or pretrained) model.")
    parser.add_argument("--model", type=str, help="Path to your trained model.pt")
    parser.add_argument("--pretrained", action="store_true",
                         help="Use general-purpose ImageNet model instead of a custom one")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to a folder of images")
    parser.add_argument("--top_k", type=int, default=3, help="Number of predictions to show per image")
    args = parser.parse_args()

    if not args.model and not args.pretrained:
        parser.error("Specify --model path/to/model.pt (your trained model) or --pretrained (general model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrained:
        model, classes = load_pretrained_model(device)
    else:
        model, classes = load_custom_model(args.model, device)

    transform = get_transform()
    images = collect_images(args)

    for image_path in images:
        try:
            results = tag_image(image_path, model, classes, transform, device, args.top_k)
        except Exception as e:
            print(f"{image_path.name}: ERROR ({e})")
            continue

        tags = ", ".join(f"{label} ({prob * 100:.1f}%)" for label, prob in results)
        print(f"{image_path.name}: {tags}")


if __name__ == "__main__":
    main()
