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
import torch.fft

def apply_fda(src_img, trg_img, beta=0.05):
    """
    Intercambia las bajas frecuencias del espectro de amplitud entre dos imágenes.
    src_img: imagen a la que se le cambia el 'estilo'.
    trg_img: imagen que provee el nuevo espectro de amplitud.
    """
    # FFT en las dimensiones espaciales (H, W)
    fft_src = torch.fft.fftn(src_img, dim=(-2, -1))
    fft_trg = torch.fft.fftn(trg_img, dim=(-2, -1))

    # Descomposición en amplitud y fase
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)

    # Definición de la ventana de baja frecuencia (L en el paper FDA)
    c, h, w = src_img.shape
    b = int(np.floor(min(h, w) * beta))
    
    # Reemplazo del centro del espectro (estilo)
    # Se modifican las esquinas porque en torch.fft la frecuencia 0 está en [0,0]
    amp_src[:, :b, :b] = amp_trg[:, :b, :b]
    amp_src[:, :b, -b:] = amp_trg[:, :b, -b:]
    amp_src[:, -b:, :b] = amp_trg[:, -b:, :b]
    amp_src[:, -b:, -b:] = amp_trg[:, -b:, -b:]

    # Reconstrucción con la fase original
    fft_src_aug = amp_src * torch.exp(1j * pha_src)
    src_aug = torch.fft.ifftn(fft_src_aug, dim=(-2, -1)).real
    
    return src_aug


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
parser.add_argument("--loss_type", type=str, default="soft", choices=["soft", "mse", "mae", "smooth_l1"])
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
parser.add_argument("--ipw", action="store_true", help="Inverse Probability Weighting por distribución de edad")
args = parser.parse_args()

# CONSTANTES FIJAS
BATCH_SIZE = 32
NUM_EPOCHS = 200
MAX_LR = args.lr
WEIGHT_DECAY = 1e-4
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

# Para reanudación lo más consistente posible
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# RUTAS AUTOMÁTICAS
# Variables de directorios (cerca de la línea 80)
BASE_PROJECT = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"
DATA_DIRS = {
    "zscore": os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/quasiraw/processed_final_SOLID_V2_Zscore_5"),
    "p99": os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/quasiraw/processed_final_SOLID_V2_P1_P99")
}

TRAIN_TXT = os.path.join(os.path.dirname(__file__), "../IDs/final_combined/v2_saneada/train_ids.txt")
VAL_TXT = os.path.join(os.path.dirname(__file__), "../IDs/final_combined/v2_saneada/val_ids.txt")

# ==============================================================================
# 2. UTILIDADES
# ==============================================================================
class DualLogger(object):
    """
    Interceptor de flujos que redirige la salida estándar (stdout/stderr) 
    a la terminal y a un archivo físico simultáneamente de forma atómica.
    """
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
# 2b. CHECKPOINTING ROBUSTO
# ==============================================================================

def get_rng_states():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None
    return state


def set_rng_states(state):
    if state is None:
        return
    
    if "torch" in state and state["torch"] is not None:
        # Forzamos la transferencia del tensor a CPU
        torch.set_rng_state(state["torch"].cpu())
        
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
        
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        # El estado de CUDA es una lista de tensores. Iteramos y mapeamos a CPU.
        cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_training_checkpoint(
    ckpt_path,
    model,
    optimizer,
    scheduler,
    epoch,
    next_batch_idx,
    best_mae,
    epochs_without_improvement,
    best_model_path,
    train_loss_accum=0.0,
    train_mae_accum=0.0,
    train_count=0,
):
    checkpoint = {
        "epoch": epoch,
        "next_batch_idx": next_batch_idx,  # batch a ejecutar al reanudar
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mae": best_mae,
        "epochs_without_improvement": epochs_without_improvement,
        "best_model_path": best_model_path,
        "rng_states": get_rng_states(),
        "train_loss_accum": train_loss_accum,
        "train_mae_accum": train_mae_accum,
        "train_count": train_count,
    }
    atomic_torch_save(checkpoint, ckpt_path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def build_dataloader(dataset, mode, epoch):
    g = torch.Generator()
    offset = 0 if mode == "train" else 100000
    g.manual_seed(SEED + offset + epoch)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
# ==============================================================================
# 3. DATASET
# ==============================================================================
class BrainAgeDataset(Dataset):
    def __init__(self, ids_file, plane, mode='train', fda_prob=0.8, fda_beta=0.05):
        self.plane = plane
        self.mode = mode
        self.fda_prob = fda_prob
        self.fda_beta = fda_beta
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
            "002_S_0413_I1221051",
            "002_S_0413_I863056",
            "sub-426134424341", "126_S_6559_I1509217", "PT030_OpenNeuro_ds005264_sub-07", "PT030_OpenNeuro_ds005264_sub-21",
            "PT030_OpenNeuro_ds003684_sub-14", "PT030_OpenNeuro_ds004815_sub-03", "PT030_OpenNeuro_ds005551_sub-11",
            "PT030_OpenNeuro_ds005216_sub-67", "PT030_OpenNeuro_ds005366_sub-099", "PT030_OpenNeuro_ds005264_sub-11",
            "sub-121916223122", "009_S_0751_I1485270", "PT030_OpenNeuro_ds003592_sub-265", "126_S_6559_I1356105",
            "PT030_OpenNeuro_ds004440_sub-45", "PT030_OpenNeuro_ds005216_sub-61", "sub-411578175554", 
            "sub-528_ses-wave2", "126_S_6559_I1038586"
        }

        with open(ids_file, "r") as f:
            raw_ids = [line.strip() for line in f.readlines() if line.strip()]

        # 1. Base Train/Val (txt originales): Exclusión estricta de dominios OOD, FOMO y Lista Negra.
        # "PT" corresponde a los sujetos de FOMO y "CP" a LongCOVID.
        self.ids = [
            sid for sid in raw_ids 
            if "OAS" not in sid 
            and "RRIB" not in sid 
            and "CERMEP" not in sid 
            and not sid.startswith("CP") 
            and not sid.startswith("PT")
            and sid not in blacklisted_ids
        ]

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
        """Enrutador determinista exclusivo para espacio Min-Max P1-P99."""
        # Estructura jerárquica real: DATA_DIR / ID_SUJETO / t1_tensors / PLANO / tensor.pt
        ruta_p99 = os.path.join(DATA_DIRS["p99"], str(subj_id), "t1_tensors", self.plane, f"{subj_id}.pt")
        
        if os.path.exists(ruta_p99):
            return ruta_p99
            
        # Al eliminar el bucle for, es matemáticamente imposible que el puntero apunte a Z-score
        raise FileNotFoundError(f"Error crítico de consistencia: El ID {subj_id} no existe en el dominio P1-P99 ({ruta_p99}).")

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
                # 1. Aumentaciones espaciales y ruido
                if torch.rand(1) < 0.5: img = TF.hflip(img)
                if torch.rand(1) < 0.5:
                    angle = float(torch.empty(1).uniform_(-AUG_ROT, AUG_ROT))
                    img = TF.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=0)
                if torch.rand(1) < 0.5:
                    img = img + (torch.randn_like(img) * AUG_NOISE_SIGMA)

                # 1.5. Aumentaciones de PSF (Nitidez y Suavizado aleatorio canal por canal)
                if torch.rand(1) < 0.3:
                    kernel_size = int(np.random.choice([3, 5]))
                    sigma = float(torch.empty(1).uniform_(0.1, 1.1))
                    img = torch.cat([
                        TF.gaussian_blur(img[c:c+1], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]) 
                        for c in range(img.shape[0])
                    ], dim=0)
                
                if torch.rand(1) < 0.3:
                    sharp_factor = float(torch.empty(1).uniform_(1.5, 3.0))
                    img = torch.cat([
                        TF.adjust_sharpness(img[c:c+1], sharp_factor) 
                        for c in range(img.shape[0])
                    ], dim=0)
                
                # 2. Fourier Domain Augmentation (FDA)
                # Seleccionamos un 'donante' de amplitud aleatorio del mismo dataset
                if torch.rand(1) < self.fda_prob:
                    try:
                        random_idx = np.random.randint(0, len(self.ids))
                        trg_sample = torch.load(self._get_path(self.ids[random_idx]), weights_only=False)
                        trg_img = trg_sample["image"].float()
                        img = apply_fda(img, trg_img, beta=self.fda_beta)
                    except Exception:
                        pass # Si falla el donante, seguimos con la imagen original sin FDA

            # La estandarización topológica (Padding/Crop) ya fue aplicada offline 
            # por harmonize_latent_space.py. El tensor ya viene con la forma correcta.

            # .clone() sigue siendo crítico si el tensor sufrió un TF.affine durante el augmentation
            return img.clone(), label, torch.tensor(age, dtype=torch.float32)
            
        except RuntimeError as e:
            # Reconstrucción de dimensiones estrictas basadas en espacio MNI152 (182x218x182)
            if self.plane == "axial":
                target_h, target_w = 182, 218
            elif self.plane == "coronal":
                target_h, target_w = 182, 182
            else: # sagittal
                target_h, target_w = 218, 182
                
            print(f"[!] Warning: Fallo al cargar tensor ({e}). Inyectando tensor nulo de {target_h}x{target_w}.")
            return torch.zeros((INPLACE, target_h, target_w)), torch.zeros(NUM_CLASSES), torch.tensor(0.0)

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

    log_filepath = os.path.join(experiment_dir, f"console_log_{plane_id}.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = DualLogger(log_filepath, original_stdout)
    sys.stderr = DualLogger(log_filepath, original_stderr)

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
        "SCHEDULER_TYPE": "ReduceLROnPlateau",
        "SUFFIX": args.suffix,
        "IPW": args.ipw,
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

    train_ds = BrainAgeDataset(TRAIN_TXT, plane, 'train')
    val_ds   = BrainAgeDataset(VAL_TXT, plane, 'val')

    # --- IPW: CÁLCULO DE PESOS POR DISTRIBUCIÓN DE EDAD ---
    ipw_weights_tensor = None
    if args.ipw:
        print("[IPW] Calculando distribución de edades en train (Estabilizado)...")
        from scipy.ndimage import gaussian_filter1d
        
        train_ages_list = []
        failed_loads = 0
        last_error = None
        
        for sid in train_ds.ids:
            try:
                sample = torch.load(train_ds._get_path(sid), map_location='cpu', weights_only=False)
                age_raw = sample["age"]
                age = float(age_raw.item() if isinstance(age_raw, torch.Tensor) else age_raw)
                train_ages_list.append(age)
            except Exception as e:
                failed_loads += 1
                last_error = e

        if not train_ages_list:
            sys.exit(f"[CRITICAL] El pipeline IPW colapsó: No se pudo cargar NINGÚN tensor. Error: {last_error}")
            
        if failed_loads > 0:
            print(f"[!] IPW Warning: Falló la carga de {failed_loads} tensores (Sujetos huérfanos).")

        age_arr = np.array(train_ages_list)
        bins = np.round(age_arr).astype(int).clip(0, NUM_CLASSES - 1)
        raw_counts = np.bincount(bins, minlength=NUM_CLASSES).astype(float)
        
        # 1. Suavizado espacial de la densidad (KDE aproximado)
        # Comparte masa de probabilidad con +- 2 años de ventana
        smoothed_counts = gaussian_filter1d(raw_counts, sigma=2.0)
        
        # 2. Laplace Smoothing (alpha) en espacio de densidad
        alpha = 0.05
        freq = smoothed_counts / smoothed_counts.sum()
        stabilized_probs = freq + alpha
        
        # 3. Inversión
        w_ipw = 1.0 / stabilized_probs
        
        # 4. Truncamiento defensivo (Clipping)
        # Previene que el top 5% de edades más raras monopolicen el gradiente
        p95_threshold = np.percentile(w_ipw, 95)
        w_ipw = np.clip(w_ipw, a_min=None, a_max=p95_threshold)
        
        # 5. Normalización para preservar norma del lote
        w_ipw = w_ipw / w_ipw.mean()
        
        ipw_weights_tensor = torch.tensor(w_ipw, dtype=torch.float32).to(DEVICE)
        print(f"[IPW] OK. Sujetos válidos: {len(train_ages_list)} | Rango Pesos: [{w_ipw.min():.3f}, {w_ipw.max():.3f}]")
        writer.add_histogram("IPW/Age_Weights", ipw_weights_tensor, 0)

    n_classes = NUM_CLASSES if args.loss_type == 'soft' else 1

    model = GlobalLocalBrainAge(
        inplace=INPLACE, patch_size=PATCH_SIZE, step=STEP, nblock=args.nblock,
        backbone=args.backbone, num_classes=n_classes, drop_rate=args.drop_rate
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    if args.loss_type == 'soft':
        # Para poder escalar por sujeto con IPW, necesitamos el vector crudo de pérdidas
        if args.ipw:
            criterion = nn.KLDivLoss(reduction='none')
        else:
            criterion = nn.KLDivLoss(reduction='batchmean')
    elif args.loss_type == 'mse':
        criterion = nn.MSELoss(reduction='none') if args.ipw else nn.MSELoss()
    elif args.loss_type == 'mae':
        criterion = nn.L1Loss(reduction='none') if args.ipw else nn.L1Loss()
    elif args.loss_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss(reduction='none') if args.ipw else nn.SmoothL1Loss()

    best_mae = float('inf')
    best_model_path = ""
    epochs_without_improvement = 0
    start_epoch = 0
    start_batch_idx = 0
    resumed_from_checkpoint = False

    latest_ckpt_path = os.path.join(experiment_dir, f"checkpoint_latest_{plane_id}.pt")
    if os.path.exists(latest_ckpt_path):
        resumed_from_checkpoint = True
        print(f"[*] Detectado checkpoint previo: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint.get("next_batch_idx", 0)
        best_mae = checkpoint["best_mae"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        best_model_path = checkpoint.get("best_model_path", "")
        set_rng_states(checkpoint.get("rng_states", None))

        print(
            f"[*] Resumiendo desde época {start_epoch+1}/{NUM_EPOCHS}, "
            f"batch {start_batch_idx}. Mejor MAE histórico: {best_mae:.4f}"
        )

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_dl = build_dataloader(train_ds, mode="train", epoch=epoch)
        val_dl   = build_dataloader(val_ds, mode="val", epoch=epoch)

        model.train()

        # Restaurar acumuladores si reanudamos una época a la mitad
        if epoch == start_epoch and start_batch_idx > 0 and resumed_from_checkpoint:
            train_loss_accum = checkpoint.get("train_loss_accum", 0.0)
            train_mae_accum = checkpoint.get("train_mae_accum", 0.0)
            train_count = checkpoint.get("train_count", 0)
            print(
                f"[*] Reanudando acumuladores de train en época {epoch+1}: "
                f"loss_accum={train_loss_accum:.4f}, "
                f"mae_accum={train_mae_accum:.4f}, "
                f"count={train_count}"
            )
        else:
            train_loss_accum = 0.0
            train_mae_accum = 0.0
            train_count = 0

        grads_list = []

        for i, (imgs, labels, ages_real) in enumerate(train_dl):
            if epoch == start_epoch and i < start_batch_idx:
                continue
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
                    if args.ipw and ipw_weights_tensor is not None:
                        # KLDiv devuelve (B, C). Sumamos sobre las clases (dim=1) para obtener el error escalar per_sample
                        per_sample = criterion(F.log_softmax(out, dim=1), labels).sum(dim=1)
                        age_idx = ages_real.round().long().clamp(0, NUM_CLASSES - 1)
                        sample_w = ipw_weights_tensor[age_idx]
                        loss += w * (per_sample * sample_w).mean()
                    else:
                        loss += w * criterion(F.log_softmax(out, dim=1), labels)
                    if idx_out > 0: batch_preds_list.append(decode_age(out))
                else:
                    out_flat = out.view(-1)
                    if args.ipw and ipw_weights_tensor is not None:
                        per_sample = criterion(out_flat, ages_real)
                        age_idx = ages_real.round().long().clamp(0, NUM_CLASSES - 1)
                        sample_w = ipw_weights_tensor[age_idx]
                        loss += w * (per_sample * sample_w).mean()
                    else:
                        loss += w * criterion(out_flat, ages_real)
                    if idx_out > 0: batch_preds_list.append(out_flat)

            loss.backward()
            
            # --- MITIGACIÓN DE EXPLOSIÓN DE GRADIENTES POR IPW ---
            # Acotamos la norma L2 global de los gradientes a 1.0 antes de la actualización
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if i == len(train_dl) - 1:
                grads_list = [p.grad for p in model.parameters() if p.grad is not None]

            optimizer.step()
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
                        batch_loss = criterion(F.log_softmax(out_head, dim=1), labels)
                        # Si IPW activó reduction='none', colapsamos el tensor a escalar manualmente para validación
                        loss_val_batch += batch_loss.sum(dim=1).mean() if args.ipw else batch_loss
                    else:
                        batch_loss = criterion(out_head.view(-1), ages_real)
                        loss_val_batch += batch_loss.mean() if args.ipw else batch_loss

                val_loss_accum += (loss_val_batch.item() / len(all_outputs)) * imgs.size(0)
                val_count += imgs.size(0)

        epoch_train_loss = train_loss_accum / train_count
        epoch_val_loss = val_loss_accum / val_count
        epoch_train_mae = train_mae_accum / train_count
        epoch_val_mae = val_mae_accum / val_count
        gen_gap = epoch_val_mae - epoch_train_mae
        
        scheduler.step(epoch_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
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

        # 1. EVALUACIÓN DE MEJORA (Early Stopping)
        is_best = epoch_val_mae < best_mae
        if is_best:
            best_mae = epoch_val_mae
            epochs_without_improvement = 0
            best_model_path = os.path.join(experiment_dir, f"best_model_{plane_id}.pt")
            # El "Mejor Modelo" sigue guardando solo la red para inferencia clínica liviana
            torch.save(model.state_dict(), best_model_path)
            print(f"🟢 RECORD: {best_mae:.4f} -> Saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"   Sin mejora. Early stopping: {epochs_without_improvement}/{args.patience}")

        # 2. CHECKPOINTING DE TOLERANCIA A FALLOS (Todos los steps)
        # Pasamos 0 en los acumuladores porque el próximo reinicio será una época nueva
        save_training_checkpoint(
            ckpt_path=latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            next_batch_idx=0,
            best_mae=best_mae,
            epochs_without_improvement=epochs_without_improvement,
            best_model_path=best_model_path,
            train_loss_accum=0.0,
            train_mae_accum=0.0,
            train_count=0,
        )

        if epoch == start_epoch:
            start_batch_idx = 0

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
    print(f"Log de Consola:  {log_filepath}") # Podés sumar esto al reporte
    print("="*85 + "\n")

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    writer.close()

if __name__ == "__main__":
    for p in args.planes:
        train_routine(p)
