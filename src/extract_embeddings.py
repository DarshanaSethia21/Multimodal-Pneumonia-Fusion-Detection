import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_frozen_resnet18(device: str = "cpu") -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def extract_embeddings_for_split(clinical_csv_path: str, out_dir: str = "data/processed",
                                  split_name: str = "train", batch_size: int = 32,
                                  device: str = "cpu") -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(clinical_csv_path)

    model = load_frozen_resnet18(device)

    embeddings = []
    batch_imgs = []

    def flush_batch():
        if not batch_imgs:
            return []
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
        return out.cpu().numpy().tolist()

    for i, path in enumerate(df["image_path"]):
        img = Image.open(path).convert("L")
        img_tensor = IMAGE_TRANSFORM(img)
        batch_imgs.append(img_tensor)

        if len(batch_imgs) == batch_size:
            embeddings.extend(flush_batch())
            batch_imgs = []

        if (i + 1) % 200 == 0:
            print(f"  [{split_name}] processed {i + 1}/{len(df)} images")

    embeddings.extend(flush_batch())

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = df["label"].to_numpy()

    emb_path = os.path.join(out_dir, f"{split_name}_embeddings.npy")
    label_path = os.path.join(out_dir, f"{split_name}_labels.npy")
    np.save(emb_path, embeddings)
    np.save(label_path, labels)

    print(f"{split_name}: saved {embeddings.shape} embeddings to {emb_path}")
    return emb_path


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        csv_path = f"data/raw/{split}_clinical.csv"
        if os.path.exists(csv_path):
            extract_embeddings_for_split(csv_path, split_name=split)
        else:
            print(f"Skipping {split}: {csv_path} not found (run clinical_features.py first)")