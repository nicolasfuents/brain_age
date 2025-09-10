# train.py — Entrena axial, coronal y sagital. Ensemble opcional.

# Entrena los tres planos y luego hace ensemble
# python train.py --ensemble

# Entrena solo axial y coronal, sin ensemble
# python train.py --planes axial coronal

# Solo ensemble (si ya entrenaste modelos)
# python train.py --ensemble --planes axial coronal sagittal

# Ejemplo de uso con todo activado
# python train.py --ensemble --gaussian-noise


import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.transforms import functional as TF

# Importar desde root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
PATIENCE = 10
SEED = 42
PATCH_SIZE = 64
STEP = 32
NBLOCK = 8
BACKBONE = "vgg16"
INPLACE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------
# DATASET
# -----------------------------
class BrainAgeDataset(Dataset):
    def __init__(self, ids_file, data_dir, use_noise=False):
        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.data_dir = data_dir
        self.use_noise = use_noise

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        sample = torch.load(os.path.join(self.data_dir, f"{id}.pt"))
        image = sample["image"]
        age = sample["age"]

        # Augmentación
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
        if torch.rand(1).item() < 0.5:
            angle = torch.empty(1).uniform_(-5, 5).item()
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if self.use_noise and torch.rand(1).item() < 0.5:
            noise = torch.randn_like(image) * 0.01
            image = image + noise

        return image, age


# -----------------------------
# ENTRENAMIENTO
# -----------------------------
def train_model_for_plane(plane, timestamp, use_noise):
    print(f"\n🚀 Entrenando plano: {plane.upper()}", flush=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, f"../../data/processed/{plane}")
    ID_FILE = os.path.join(BASE_DIR, f"../IDs/train_ids_original.txt")
    MODEL_PATH = os.path.join(BASE_DIR, f"../models/model_{plane}_{timestamp}.pt")
    TBOARD_LOGDIR = os.path.join(BASE_DIR, f"../runs/{plane}_{timestamp}")

    dataset = BrainAgeDataset(ID_FILE, DATA_DIR, use_noise)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GlobalLocalBrainAge(inplace=INPLACE, patch_size=PATCH_SIZE,
                                 step=STEP, nblock=NBLOCK, backbone=BACKBONE).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    writer = SummaryWriter(TBOARD_LOGDIR)

    best_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        preds_all = []
        targets_all = []
        epoch_loss = 0.0

        for images, ages in dataloader:
            images = images.to(DEVICE)
            ages = ages.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            preds = outputs[0]
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds_all.append(preds.detach().cpu())
            targets_all.append(ages.detach().cpu())

        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)
        mae_epoch = torch.mean(torch.abs(preds_all - targets_all)).item()
        avg_loss = epoch_loss / len(dataset)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("MAE/train", mae_epoch, epoch)

        print(f"Época {epoch+1:02d} - Loss: {avg_loss:.2f} - MAE: {mae_epoch:.2f}", flush=True)

        if mae_epoch < best_mae:
            best_mae = mae_epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Modelo guardado")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping ({PATIENCE} épocas sin mejora)")
                break

    writer.close()
    return best_mae


# -----------------------------
# ENSEMBLE VALIDACIÓN
# -----------------------------
def evaluate_ensemble(planes, timestamp, maes_dict):
    print(f"\nEvaluando ensemble en validación ({', '.join(planes)})", flush=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAL_FILE = os.path.join(BASE_DIR, f"../IDs/val_ids.txt")

    with open(VAL_FILE, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]

    age_dict = {}
    pred_dict = {id: [] for id in val_ids}

    for plane in planes:
        DATA_DIR = os.path.join(BASE_DIR, f"../../data/processed/{plane}")
        MODEL_PATH = os.path.join(BASE_DIR, f"../models/model_{plane}_{timestamp}.pt")

        model = GlobalLocalBrainAge(inplace=INPLACE, patch_size=PATCH_SIZE,
                                     step=STEP, nblock=NBLOCK, backbone=BACKBONE).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        for id in val_ids:
            sample = torch.load(os.path.join(DATA_DIR, f"{id}.pt"))
            image = sample["image"].unsqueeze(0).to(DEVICE)
            age = float(sample["age"])
            age_dict[id] = age

            with torch.no_grad():
                output = model(image)[0].item()
            pred_dict[id].append(output)

    final_preds = []
    final_ages = []
    for id in val_ids:
        mean_pred = np.mean(pred_dict[id])
        true_age = age_dict[id]
        final_preds.append(mean_pred)
        final_ages.append(true_age)

    final_preds = np.array(final_preds)
    final_ages = np.array(final_ages)
    mae = np.mean(np.abs(final_preds - final_ages))
    mse = np.mean((final_preds - final_ages) ** 2)

    result_str = f"\nMAE por plano:\n"
    for plane in planes:
        result_str += f"- {plane}: {maes_dict[plane]:.2f}\n"
    result_str += f"\nEnsemble MAE: {mae:.2f} años | MSE: {mse:.2f}\n"

    print(result_str, flush=True)

    result_file = os.path.join(BASE_DIR, f"../models/ensemble_results_{timestamp}.txt")
    with open(result_file, "w") as f:
        f.write(result_str)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planes", nargs="+", default=["axial", "coronal", "sagittal"],
                        help="Planes a entrenar (ej: axial coronal)")
    parser.add_argument("--ensemble", action="store_true", help="Evaluar ensemble")
    parser.add_argument("--gaussian-noise", action="store_true", help="Agregar ruido gaussiano (σ=0.01)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    maes = {}

    # Comenzar cronómetro
    start_time = datetime.now()

    print(f"Config: BACKBONE={BACKBONE}, PATCH={PATCH_SIZE}, PATIENCE={PATIENCE}, STEP={STEP}, NBLOCK={NBLOCK}, "
          f"INPLACE={INPLACE}, BATCH={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}\n",
          flush=True)

    for plane in args.planes:
        mae = train_model_for_plane(plane, timestamp, args.gaussian_noise)
        maes[plane] = mae

    # Fin y cálculo de duración
    end_time = datetime.now()
    duration = end_time - start_time

    print("\nMAE de entrenamiento por plano:", flush=True)
    for p in args.planes:
        print(f"  {p}: {maes[p]:.2f} años")

    print(f"\nTiempo total de entrenamiento: {str(duration)}", flush=True)

    # Mostrar modelos generados
    print("\nModelos generados en este entrenamiento:", flush=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for plane in args.planes:
        model_path = os.path.join(BASE_DIR, f"../models/model_{plane}_{timestamp}.pt")
        if os.path.exists(model_path):
            print(f" - {os.path.basename(model_path)}")
        else:
            print(f" - {os.path.basename(model_path)} (no encontrado)")

    if args.ensemble:
        evaluate_ensemble(args.planes, timestamp, maes)