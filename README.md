# Simple Image Tagger

A practical image classification system using PyTorch and transfer
learning. Tags images into classes like `cat`, `dog`, `car` — or any
custom categories you train it on.

## Why transfer learning?

Training a CNN from scratch needs millions of labeled images and days
of GPU time. This project instead starts from **MobileNetV2**,
pretrained on ImageNet (1.4M images), and only retrains the final
layer for your specific classes. Result: good accuracy from a few
hundred images per class, trainable in minutes on a laptop CPU.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Option A — Use it immediately, no training

MobileNetV2 already knows 1000 general categories (including cat,
dog, car, and hundreds more).

```bash
python predict.py --pretrained --image path/to/photo.jpg
```

```bash
python predict.py --pretrained --image_dir path/to/folder/
```

Output:
```
photo.jpg: golden retriever (91.2%), Labrador retriever (4.1%), beagle (0.8%)
```

## 3. Option B — Train on your own categories

Organize your images like this:

```
data/
    train/
        cat/    cat1.jpg  cat2.jpg  ...
        dog/    dog1.jpg  dog2.jpg  ...
        car/    car1.jpg  car2.jpg  ...
    val/
        cat/    ...
        dog/    ...
        car/    ...
```

A folder name = a class name. Aim for at least ~100 images per class
for a first pass (more is better); split ~80% into `train/` and ~20%
into `val/`.

Then train:

```bash
python train.py --data_dir data --epochs 10 --output model.pt
```

This prints validation accuracy after each epoch and saves
`model.pt` (weights + class names in one file).

Tag new images with your trained model:

```bash
python predict.py --model model.pt --image path/to/photo.jpg
python predict.py --model model.pt --image_dir path/to/folder/
```

## Tuning tips

- **Low accuracy?** Add more training images, or increase `--epochs`.
- **Slow on CPU?** Reduce `--batch_size`, or use a smaller subset
  first to sanity-check the pipeline before a full run.
- **Have a GPU?** Nothing to configure — the scripts auto-detect
  CUDA and use it if available.
- **More classes?** Just add more subfolders under `train/` and
  `val/`; no code changes needed.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains a model on your labeled image folders |
| `predict.py` | Tags new images with a trained (or the built-in pretrained) model |
| `requirements.txt` | Python dependencies |

## Extending this further

- Swap `mobilenet_v2` for a larger backbone (e.g. `resnet50`) in
  `train.py`/`predict.py` if you need higher accuracy and have more
  compute.
- Unfreeze more layers (`model.features`) and lower the learning
  rate for "fine-tuning" once your top-layer model plateaus.
- Wrap `predict.py`'s logic in a small Flask/FastAPI endpoint to
  serve tags over HTTP for a real application.
