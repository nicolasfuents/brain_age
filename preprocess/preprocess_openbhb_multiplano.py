# preprocess_openbhb_multiplano.py — versión multi-plano (axial, coronal, sagital)

import os
import sys
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from skimage.transform import resize
from tqdm import tqdm

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
DATA_ROOT = "/home/nfuentes/brain_age_project/data"
SPLITS = ["train", "val"]
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed")
PLANES = ["axial", "coronal", "sagittal"]
NUM_CORTES = 5
ID_LISTS_DIR = "/home/nfuentes/brain_age_project/scripts/IDs"

for plane in PLANES:
    os.makedirs(os.path.join(OUTPUT_DIR, plane), exist_ok=True)
os.makedirs(ID_LISTS_DIR, exist_ok=True)

resize_fn = transforms.Compose([
    transforms.Lambda(lambda x: resize(x, (160, 160), preserve_range=True, anti_aliasing=True)),
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))  # No agrega canal
])


# -------------------------------
# PROCESAMIENTO POR SPLIT
# -------------------------------
for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, split, "quasiraw")
    tsv_path = os.path.join(DATA_ROOT, split, f"{split}_labels", "participants.tsv")
    df = pd.read_csv(tsv_path, sep="\t")

    df["participant_id"] = df["participant_id"].astype(str).str.replace(",", "").str.strip()
    df["participant_id"] = df["participant_id"].apply(lambda x: x.split(".")[0] if "e" in x.lower() else x)

    id_to_age = dict(zip(df["participant_id"], df["age"]))

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    valid_ids = []

    for fname in tqdm(files, desc=f"Preprocesando {split}", file=sys.stdout):
        try:
            fpath = os.path.join(img_dir, fname)
            sub_id = fname.split("_")[0].replace("sub-", "")

            if sub_id not in id_to_age:
                print(f"[SKIP] {fname}: ID no está en participants.tsv")
                continue

            age = float(id_to_age[sub_id])
            if not np.isfinite(age) or age < 0 or age > 120:
                print(f"[SKIP] {fname}: Edad inválida ({age})")
                continue

            img = np.load(fpath)
            img = np.squeeze(img)
            if img.ndim != 3:
                raise ValueError(f"{fname}: volumen no tiene 3 dimensiones después de squeeze → {img.shape}")

            H, W, D = img.shape
            center = {"sagittal": H // 2, "coronal": W // 2, "axial": D // 2}
            half = NUM_CORTES // 2
            offsets = list(range(-half, half + 1)) if NUM_CORTES % 2 == 1 else list(range(-half, half))

            base_id = fname.replace(".npy", "")
            for plane in PLANES:
                cortes = []
                for offset in offsets:
                    if plane == "sagittal":
                        idx = center[plane] + offset
                        if idx < 0 or idx >= H: continue
                        s = img[idx, :, :]
                    elif plane == "coronal":
                        idx = center[plane] + offset
                        if idx < 0 or idx >= W: continue
                        s = img[:, idx, :]
                    elif plane == "axial":
                        idx = center[plane] + offset
                        if idx < 0 or idx >= D: continue
                        s = img[:, :, idx]

                    s = np.clip(s, 0, np.percentile(s, 99))
                    if s.shape[0] < 10 or s.shape[1] < 10:
                        with open("cortes_descartados.log", "a") as logf:
                            logf.write(f"{fname} {plane}: Corte inválido con shape {s.shape}\n")
                        raise ValueError(f"{fname} {plane}: corte demasiado pequeño: {s.shape}")
                    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
                    s_resized = resize_fn(s)
                    cortes.append(s_resized)

                if len(cortes) != NUM_CORTES:
                    print(f"[SKIP] {fname}: solo {len(cortes)} cortes válidos en plano {plane}")
                    continue

                volume = torch.stack(cortes, dim=0)  # [NUM_CORTES, 160, 160]
                torch.save({
                    "image": volume,
                    "age": torch.tensor(age, dtype=torch.float32)
                }, os.path.join(OUTPUT_DIR, plane, base_id + ".pt"))

            valid_ids.append(base_id)

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    list_path = os.path.join(ID_LISTS_DIR, f"{split}_ids.txt")
    with open(list_path, "w") as f:
        for id in valid_ids:
            f.write(id + "\n")
    print(f"Lista guardada: {list_path} ({len(valid_ids)} sujetos)")
