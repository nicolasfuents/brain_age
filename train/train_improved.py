import os
import sys
import argparse
import json
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.transforms import functional as TF

# ==============================================================================
# 0. SETUP
# ==============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from GlobalLocalTransformer_soft_labels import GlobalLocalBrainAge
except ImportError:
    sys.exit("CRITICAL: No se encuentra GlobalLocalTransformer_soft_labels. Revisá el path.")

# ==============================================================================
# 1. ARGUMENTOS
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--planes", nargs="+", default=["axial"])
parser.add_argument("--backbone", type=str, default="resnet18", help="vgg8, vgg16, resnet18, resnet34")
parser.add_argument("--loss_type", type=str, default="soft", choices=["soft", "mse", "mae"])
parser.add_argument("--sigma", type=float, default=1.0, help="Sigma para Soft Labels")
parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout en Atención")
parser.add_argument("--timestamp", type=str, default="MANUAL")
parser.add_argument("--enable-aug", action="store_true")
parser.add_argument("--nblock", type=int, default=8)
# --- NUEVO ARGUMENTO: SUFIJO PARA EVITAR COLISIONES ---
parser.add_argument("--suffix", type=str, default="", help="Sufijo opcional para el nombre del modelo/log (ej: 'purist')")
parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
parser.add_argument("--aug_rot", type=float, default=10.0, help="Grados de rotación")
parser.add_argument("--aug_noise", type=float, default=0.01, help="Sigma del ruido")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate máximo")
parser.add_argument("--patience", type=int, default=40, help="Épocas de paciencia para Early Stopping")
args = parser.parse_args()

# CONSTANTES FIJAS
BATCH_SIZE = 8
NUM_EPOCHS = 200
MAX_LR = args.lr          
WEIGHT_DECAY = 1e-4  # Ajustado a 0.0001 según tu tabla
NUM_CLASSES = 100
PATCH_SIZE = 64
STEP = 32
INPLACE = 5
AUG_ROT = args.aug_rot
AUG_NOISE_SIGMA = args.aug_noise
SEED = args.seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# RUTAS AUTOMÁTICAS
BASE_PROJECT = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"
DATA_DIR_OPENBHB = os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/processed_p99")
DATA_DIR_OASIS = os.path.join(BASE_PROJECT, "data/processed_training_olders_T1_p99")
DATA_DIR_FOMO = os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/processed_fomo_p99")
CSV_FOMO = os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/participants.csv")
TRAIN_TXT = os.path.join(os.path.dirname(__file__), "../IDs/final_combined/train_ids.txt")
VAL_TXT = os.path.join(os.path.dirname(__file__), "../IDs/final_combined/val_ids.txt")

# ==============================================================================
# 2. UTILIDADES
# ==============================================================================
def get_dataset_name(subject_id):
    """Clasifica IDs en bases de datos según patrones de texto."""
    sid = str(subject_id).strip()
    
    if 'CERMEP' in sid: return 'CERMEP'
    if 'AOMIC' in sid: return 'OpenNeuro_AOMIC'
    if '_S_' in sid and sid[0].isdigit(): return 'ADNI3'
    if sid.startswith('CP'): return 'LongCOVID'
    if 'JUK' in sid: return 'JUK'
    if sid.startswith('PT'): return 'FOMO'
    
    if sid.startswith('sub-'):
        if 'sub-CC' in sid: return 'Cam-CAN'
        if 'ses-wave' in sid: return 'OpenNeuro_2'
        return 'OpenBHB' # Por descarte

    # Extraccion dinamica del prefijo para agrupar sujetos desconocidos
    # Cortamos en el primer guion bajo o medio para aislar la raiz del ID
    patron_raiz = sid.split('_')[0].split('-')[0]
    return f'Otros (Patron: {patron_raiz})'

def generate_gaussian_label(age, sigma):
    age = min(max(age, 0), NUM_CLASSES - 1)
    x = torch.arange(NUM_CLASSES).float()
    dist = torch.exp(-0.5 * ((x - age) / sigma) ** 2)
    return dist / dist.sum()

def decode_age(logits):
    probs = F.softmax(logits, dim=1)
    x = torch.arange(probs.shape[1], device=probs.device).float()
    return (probs * x).sum(dim=1)

def calculate_rms(tensor_list):
    """Calcula el RMS (Root Mean Square) de una lista de tensores (grads o weights)."""
    sq_sum = 0.0
    count = 0
    for t in tensor_list:
        if t is not None:
            sq_sum += torch.sum(t ** 2).item()
            count += t.numel()
    return np.sqrt(sq_sum / count) if count > 0 else 0.0

# ==============================================================================
# 3. DATASET
# ==============================================================================
class BrainAgeDataset(Dataset):
    def __init__(self, ids_file, plane, mode='train'):
        self.plane = plane
        self.mode = mode
        if not os.path.exists(ids_file):
            sys.exit(f"No existe IDs file: {ids_file}")

        # --- LISTA NEGRA DE SUJETOS A EXCLUIR ---
        blacklisted_ids = {
            "PT030_OpenNeuro_ds001942_sub-04", "PT030_OpenNeuro_ds004958_sub-33", "PT030_OpenNeuro_ds004958_sub-05", 
            "PT030_OpenNeuro_ds004958_sub-04", "PT030_OpenNeuro_ds004958_sub-22", "PT030_OpenNeuro_ds004958_sub-18", 
            "PT030_OpenNeuro_ds004958_sub-31", "PT030_OpenNeuro_ds004958_sub-34", "PT030_OpenNeuro_ds006040_sub-11", 
            "PT030_OpenNeuro_ds004958_sub-07", "PT030_OpenNeuro_ds004958_sub-30", "PT030_OpenNeuro_ds004958_sub-32", 
            "PT030_OpenNeuro_ds003192_sub-05", "PT030_OpenNeuro_ds003192_sub-03", "PT030_OpenNeuro_ds003192_sub-06", 
            "PT030_OpenNeuro_ds004928_sub-02", "PT030_OpenNeuro_ds004958_sub-15", "099_S_4086_I943534", 
            "sub-426134424341", "126_S_6559_I1509217", "PT030_OpenNeuro_ds005264_sub-07", "PT030_OpenNeuro_ds005264_sub-21", 
            "PT030_OpenNeuro_ds003684_sub-14", "PT030_OpenNeuro_ds004815_sub-03", "PT030_OpenNeuro_ds005551_sub-11", 
            "PT030_OpenNeuro_ds005216_sub-67", "PT030_OpenNeuro_ds005366_sub-099", "PT030_OpenNeuro_ds005264_sub-11", 
            "sub-121916223122", "009_S_0751_I1485270", "PT030_OpenNeuro_ds003592_sub-265", "126_S_6559_I1356105", 
            "PT030_OpenNeuro_ds004440_sub-45", "PT030_OpenNeuro_ds005216_sub-61"
        }
            
        with open(ids_file, "r") as f:
            raw_ids = [line.strip() for line in f.readlines() if line.strip()]
            
        # 1. Base OpenBHB (txt originales): Filtramos OASIS3, RRIB y la Lista Negra
        base_ids = [subj_id for subj_id in raw_ids if "OAS" not in subj_id and "RRIB" not in subj_id and subj_id not in blacklisted_ids]
        
        # 2. Extracción de FOMO desde el CSV general (Master Log)
        df_all = pd.read_csv(CSV_FOMO)
        
        # Aislamos FOMO y blindamos contra OASIS, RRIB y la Lista Negra
        fomo_mask = (df_all['ID'].astype(str).str.startswith('PT')) & \
                    (~df_all['ID'].astype(str).str.contains('OAS')) & \
                    (~df_all['ID'].astype(str).str.contains('RRIB')) & \
                    (~df_all['ID'].astype(str).isin(blacklisted_ids))
        fomo_ids = df_all[fomo_mask]['ID'].tolist()
        
        # Eliminamos posibles duplicados y ordenamos lexicográficamente para reproducibilidad estricta
        fomo_ids = sorted(list(set(fomo_ids)))
        
        # Split 80/20 anclado a la semilla global
        rng = np.random.RandomState(SEED)
        rng.shuffle(fomo_ids)
        
        split_idx = int(len(fomo_ids) * 0.8)
        fomo_split = fomo_ids[:split_idx] if self.mode == 'train' else fomo_ids[split_idx:]
            
        # Fusión de los subsets
        self.ids = base_ids + fomo_split
        
        # Conteo de distribución por base de datos
        db_counts = {}
        for sid in self.ids:
            db = get_dataset_name(sid)
            db_counts[db] = db_counts.get(db, 0) + 1
            
        # Ordenamos el diccionario por cantidad de mayor a menor para reporte limpio
        db_counts_sorted = dict(sorted(db_counts.items(), key=lambda item: item[1], reverse=True))
        
        print(f"\n" + "="*50)
        print(f"DISTRIBUCIÓN DE DATOS [{self.mode.upper()}]")
        print(f"Total de sujetos: {len(self.ids)}")
        print("-" * 50)
        for db, count in db_counts_sorted.items():
            print(f"  > {db}: {count}")
        print("="*50 + "\n")

    def __len__(self): return len(self.ids)

    def _get_path(self, subj_id):
        # 1. Evaluación FOMO (Estructura jerárquica)
        p_fomo = os.path.join(DATA_DIR_FOMO, subj_id, "t1_tensors", self.plane, f"{subj_id}.pt")
        if os.path.exists(p_fomo): return p_fomo
        
        # 2. Evaluación OpenBHB (Estructura plana)
        p_bhb = os.path.join(DATA_DIR_OPENBHB, self.plane, f"{subj_id}.pt")
        if os.path.exists(p_bhb): return p_bhb
        
        # 3. Fallback OASIS (Por robustez de la función, aunque estén excluidos)
        return os.path.join(DATA_DIR_OASIS, subj_id, "t1_tensors", f"{self.plane}.pt")

    def __getitem__(self, idx):
        try:
            sample = torch.load(self._get_path(self.ids[idx]), weights_only=False)
            img = sample["image"].float()
            age_raw = sample["age"]
            age = float(age_raw.item() if isinstance(age_raw, torch.Tensor) else age_raw)
            
            if args.loss_type == 'soft':
                label = generate_gaussian_label(age, args.sigma)
            else:
                label = torch.tensor(age, dtype=torch.float32)

            # Augmentation
            if self.mode == 'train' and args.enable_aug:
                if torch.rand(1) < 0.5: img = TF.hflip(img)
                if torch.rand(1) < 0.5: 
                    angle = float(torch.empty(1).uniform_(-AUG_ROT, AUG_ROT))
                    img = TF.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=0)
                if torch.rand(1) < 0.5: 
                    img = img + (torch.randn_like(img) * AUG_NOISE_SIGMA)

            return img, label, torch.tensor(age, dtype=torch.float32)
        except Exception:
            return torch.zeros((INPLACE, 130, 170)), torch.zeros(NUM_CLASSES), torch.tensor(0.0)

# ==============================================================================
# 4. ENTRENAMIENTO
# ==============================================================================
def train_routine(plane):
    # --- GESTIÓN DE DIRECTORIOS UNIFICADA ---
    run_name = args.timestamp 
    
    experiment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{run_name}"))
    os.makedirs(experiment_dir, exist_ok=True)
    
    # --- GESTIÓN DE SUFIJO (NUEVO) ---
    # Si hay sufijo, lo agregamos al nombre del plano para diferenciar: 'axial_purist'
    plane_id = f"{plane}_{args.suffix}" if args.suffix else plane

    print(f"--- Entrenando: {plane_id} ---")
    
    config_log = {
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": MAX_LR,
        "PATIENCE": args.patience,
        "SEED": SEED,
        "PATCH_SIZE": PATCH_SIZE,
        "STEP": STEP,
        "NBLOCK": args.nblock,
        "BACKBONE": args.backbone,
        "INPLACE": INPLACE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "AUG_ROTATION_DEG": AUG_ROT,
        "AUG_NOISE_SIGMA": AUG_NOISE_SIGMA,
        "LOSS_TYPE": args.loss_type.upper(),
        "SCHEDULER_TYPE": "OneCycleLR",
        "SUFFIX": args.suffix # Logueamos el sufijo
    }
    
    if args.loss_type == 'soft':
        config_log["SIGMA"] = args.sigma
        
    print("--- HIPERPARÁMETROS (CONFIG) ---")
    print(json.dumps(config_log, indent=4))
    
    print("-" * 80)
    print(f"Ruta TensorBoard: {run_name}/{plane_id}") 
    print("-" * 80)

    # Tensorboard usa 'plane_id' para crear subcarpeta separada (ej: axial_purist)
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, plane_id))
    
    train_dl = DataLoader(BrainAgeDataset(TRAIN_TXT, plane, 'train'), 
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(BrainAgeDataset(VAL_TXT, plane, 'val'), 
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    n_classes = NUM_CLASSES if args.loss_type == 'soft' else 1
    
    model = GlobalLocalBrainAge(
        inplace=INPLACE, patch_size=PATCH_SIZE, step=STEP, nblock=args.nblock,
        backbone=args.backbone, num_classes=n_classes, drop_rate=args.drop_rate
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, 
                                              steps_per_epoch=len(train_dl), 
                                              epochs=NUM_EPOCHS, pct_start=0.3)
    
    if args.loss_type == 'soft':
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'mae':
        criterion = nn.L1Loss()

    best_mae = float('inf')
    best_model_path = ""
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        train_mae_accum = 0.0
        train_count = 0
        grads_list = []
        
        for i, (imgs, labels, ages_real) in enumerate(train_dl):
            # 1. MANDAR DATOS A LA GPU Y LIMPIAR GRADIENTES (¡Esto se había borrado!)
            imgs, labels, ages_real = imgs.to(DEVICE), labels.to(DEVICE), ages_real.to(DEVICE)
            optimizer.zero_grad()

            # 2. LOGUEAR IMÁGENES AL TENSORBOARD
            if i == 0 and epoch % 5 == 0:  # Solo el primer batch cada 5 épocas
                # Selecciona el slice central del parche inplace (idx 2) y agrega canal (B, 1, H, W)
                img_grid = imgs[:4, 2:3, :, :] 
                writer.add_images('Inputs/Augmented_Center_Slice', img_grid, epoch)
            
            # 3. FORWARD PASS
            outputs = model(imgs)
            loss = 0
            w_losses = [1.0] + [0.5] * (len(outputs) - 1)
            batch_preds_list = []

            for idx_out, (out, w) in enumerate(zip(outputs, w_losses)):
                if args.loss_type == 'soft':
                    loss += w * criterion(F.log_softmax(out, dim=1), labels)
                    if idx_out > 0: batch_preds_list.append(decode_age(out))
                else:
                    out_flat = out.view(-1)
                    loss += w * criterion(out_flat, ages_real)
                    if idx_out > 0: batch_preds_list.append(out_flat)

            loss.backward()
            if i == len(train_dl) - 1:
                grads_list = [p.grad for p in model.parameters() if p.grad is not None]

            optimizer.step()
            scheduler.step()
            train_loss_accum += loss.item() * imgs.size(0)
            
            if len(batch_preds_list) > 0:
                avg_preds = torch.stack(batch_preds_list).mean(dim=0)
            else:
                avg_preds = decode_age(outputs[0]) if args.loss_type == 'soft' else outputs[0].view(-1)
                
            train_mae_accum += torch.sum(torch.abs(avg_preds - ages_real)).item()
            train_count += imgs.size(0)

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss_accum = 0.0
        val_mae_accum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for imgs, labels, ages_real in val_dl:
                imgs, labels, ages_real = imgs.to(DEVICE), labels.to(DEVICE), ages_real.to(DEVICE)
                outputs = model(imgs)
                
                # --- NUEVO: Registro de Distribución Softmax ---
                # Condición: Solo primer batch (val_count == 0), modelo soft, cada 5 épocas
                if val_count == 0 and args.loss_type == 'soft' and epoch % 5 == 0:
                    # outputs[0] es la salida de la rama global. [0] toma el primer sujeto del batch.
                    prob_dist = F.softmax(outputs[0][0], dim=0) 
                    writer.add_histogram('Predictions/Softmax_Distribution', prob_dist, epoch)
                
                # ENSEMBLE: Global + Locales
                batch_preds_list = []
                all_outputs = outputs 
                
                for out_head in all_outputs:
                    if args.loss_type == 'soft':
                        batch_preds_list.append(decode_age(out_head))
                    else:
                        batch_preds_list.append(out_head.view(-1))
                
                avg_preds = torch.stack(batch_preds_list).mean(dim=0)
                val_mae_accum += torch.sum(torch.abs(avg_preds - ages_real)).item()
                
                loss_val_batch = 0
                for out_head in all_outputs:
                    if args.loss_type == 'soft':
                        loss_val_batch += criterion(F.log_softmax(out_head, dim=1), labels)
                    else:
                        loss_val_batch += criterion(out_head.view(-1), ages_real)
                
                val_loss_accum += (loss_val_batch.item() / len(all_outputs)) * imgs.size(0)
                val_count += imgs.size(0)

        epoch_train_loss = train_loss_accum / train_count
        epoch_val_loss = val_loss_accum / val_count
        epoch_train_mae = train_mae_accum / train_count
        epoch_val_mae = val_mae_accum / val_count
        gen_gap = epoch_val_mae - epoch_train_mae
        current_lr = scheduler.get_last_lr()[0]
        grad_rms = calculate_rms(grads_list)
        weight_rms = calculate_rms([p.data for p in model.parameters()])

        writer.add_scalars("Loss", {"Train": epoch_train_loss, "Val": epoch_val_loss}, epoch)
        writer.add_scalars("MAE", {"Train": epoch_train_mae, "Val": epoch_val_mae}, epoch)
        writer.add_scalar("Generalization/Gap", gen_gap, epoch)
        writer.add_scalar("Hyperparams/Learning_Rate", current_lr, epoch)
        writer.add_scalar("Diagnostics/Gradient_RMS", grad_rms, epoch)
        writer.add_scalar("Diagnostics/Weight_RMS", weight_rms, epoch)

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
             for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        print(f"Ep {epoch+1:03d} | Train MAE: {epoch_train_mae:.2f} | Val MAE: {epoch_val_mae:.4f} | Loss Tr/Val: {epoch_train_loss:.2f}/{epoch_val_loss:.2f}")

        if epoch_val_mae < best_mae:
            best_mae = epoch_val_mae
            epochs_without_improvement = 0
            # GUARDADO: Si tiene sufijo, lo agrega al nombre del archivo
            filename = f"best_model_{plane_id}.pt"
            save_path = os.path.join(experiment_dir, filename)
            torch.save(model.state_dict(), save_path)
            best_model_path = save_path
            print(f"   RECORD: {best_mae:.4f} -> Saved to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"   Sin mejora. Early stopping: {epochs_without_improvement}/{args.patience}")
            if epochs_without_improvement >= args.patience:
                print(f"--- EARLY STOPPING ALCANZADO EN LA ÉPOCA {epoch+1} ---")
                break

    # Registrar HParams con la métrica objetivo final
    writer.add_hparams(
        hparam_dict=config_log,
        metric_dict={'hparam/best_val_mae': best_mae}
    )
    
    tb_path = os.path.abspath(os.path.join(experiment_dir, plane_id))
    
    print("\n" + "="*85)
    print("REPORTE FINAL DE ENTRENAMIENTO")
    print("-" * 85)
    print(f"Plano:           {plane_id.upper()}")
    print(f"Mejor MAE (Val): {best_mae:.4f}")
    print(f"Ruta del Modelo: {best_model_path}")
    print(f"Directorio TB:   {tb_path}")
    print("="*85 + "\n")
    
    writer.close()

if __name__ == "__main__":
    for p in args.planes:
        train_routine(p)