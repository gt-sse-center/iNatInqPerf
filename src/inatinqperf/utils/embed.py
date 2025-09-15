import os, numpy as np, torch
from typing import List, Tuple
from datasets import load_from_disk, Dataset, Features, Value, Sequence
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def pilify(img):
    if isinstance(img, Image.Image): return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")

def embed_images(raw_dir: str, model_id: str, batch: int) -> Tuple[Dataset, np.ndarray, list, list]:
    ds = load_from_disk(raw_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    proc = CLIPProcessor.from_pretrained(model_id)

    imgs = [pilify(r["image"]) for r in ds]
    all_emb, ids, labels = [], [], []
    for i in range(0, len(imgs), batch):
        batch_imgs = imgs[i:i+batch]
        with torch.no_grad():
            inputs = proc(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_emb.append(feats.cpu().numpy().astype("float32"))
        ids.extend([i+j for j in range(len(batch_imgs))])
        labels.extend([int(ds[i+j].get("labels", ds[i+j].get("label", 0))) for j in range(len(batch_imgs))])

    X = np.concatenate(all_emb, axis=0)
    return ds, X, ids, labels

def to_hf_dataset(X: np.ndarray, ids, labels) -> Dataset:
    feats = Features({
        "id": Value("int64"),
        "label": Value("int32"),
        "embedding": Sequence(Value("float32"), length=int(X.shape[1])),
    })
    return Dataset.from_dict({
        "id": [int(i) for i in ids],
        "label": [int(y) for y in labels],
        "embedding": [row.tolist() for row in X],
    }, features=feats)

def embed_text(queries: List[str], model_id: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    proc = CLIPProcessor.from_pretrained(model_id)
    with torch.no_grad():
        inputs = proc(text=queries, return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy().astype("float32")