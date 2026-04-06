# Usage: 
# python inference_ood_unified.py --all
# python inference_ood_unified.py --rrib --oasis
import os
import sys
import argparse
import warnings
import torch
import numpy as np

# Supresión global de warnings (FutureWarning de PyTorch y UserWarning de Matplotlib)
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from tqdm import tqdm

# ==============================================================================
# 0. SETUP ARQUITECTURAL
# ==============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from GlobalLocalTransformer import GlobalLocalBrainAge
except ImportError:
    sys.exit("CRITICAL: No se pudo importar GlobalLocalTransformer.")

parser = argparse.ArgumentParser(description="Unified OOD Inference Pipeline")
parser.add_argument("--intecnus", action="store_true", help="Inferencia en cohorte INTECNUS (BRAVO)")
parser.add_argument("--oasis", action="store_true", help="Inferencia en cohorte OASIS-3")
parser.add_argument("--rrib", action="store_true", help="Inferencia en cohorte RRIB")
parser.add_argument("--all", action="store_true", help="Ejecutar todas las cohortes OOD")
args = parser.parse_args()

if args.all:
    args.intecnus = args.oasis = args.rrib = True

# ==============================================================================
# 1. CONSTANTES Y RUTAS GLOBALES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models/final_8k")
PLOTS_DIR = os.path.join(BASE_DIR, "../plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

BASE_PROJECT = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 64; STEP = 32; INPLACE = 5
# Se mantiene 64 para saturación óptima de VRAM en inferencia
BATCH_SIZE = 64 

# ==============================================================================
# 2. DICCIONARIO TOPOLÓGICO OOD
# Define la estructura de parseo para cada dataset
# ==============================================================================
OOD_CONFIG = {
    "INTECNUS": {
        "path": os.path.join(BASE_PROJECT, "data/DB_INTECNUS_BRAVO/processed"),
        "structure": "nested", # subj_id/t1_tensors/plane/subj_id.pt
        "filter": ""
    },
    "OASIS3": {
        "path": os.path.join(BASE_PROJECT, "data/OASIS3/MR_867_HC/processed"),
        "structure": "nested", 
        "filter": "OAS"
    },
    "RRIB": {
        "path": os.path.join(BASE_PROJECT, "data/DB_Lautaro_quasiraw/processed_p99"),
        "structure": "flat",   # plane/subj_id.pt
        "filter": "RRIB"
    }
}

BEST_MODELS = {
    "cor_mse": {
        "file": "../TRIPLANE_ALL_MSE_NO_OASIS_361984/best_model_coronal_coronal_all_mse.pt", 
        "backbone": "vgg16", "plane_type": "coronal", "label": "Coronal MSE"
    },
    "sag_mse": {
        "file": "../TRIPLANE_ALL_MSE_NO_OASIS_363402/best_model_sagittal_sagittal_all_mse.pt", 
        "backbone": "vgg16", "plane_type": "sagittal", "label": "Sagittal MSE"
    },
    "axi_mse": {
        "file": "../TRIPLANE_ALL_MSE_NO_OASIS_363402/best_model_axial_axial_all_mse.pt", 
        "backbone": "vgg16", "plane_type": "axial", "label": "Axial MSE"
    }
}

# ==============================================================================
# 3. MOTOR DE DATOS UNIFICADO
# ==============================================================================
class UnifiedInferenceDataset(Dataset):
    def __init__(self, cfg, plane):
        self.plane = plane
        self.data_dir = cfg["path"]
        self.structure = cfg["structure"]
        self.filter = cfg["filter"]
        self.ids = []
        
        if not os.path.exists(self.data_dir):
            print(f"[WARN] Ruta no encontrada: {self.data_dir}")
            return

        # Parseo algorítmico según topología
        if self.structure == "nested":
            for d in os.listdir(self.data_dir):
                if self.filter and self.filter not in d: continue
                subj_path = os.path.join(self.data_dir, d)
                if os.path.isdir(subj_path):
                    tensor_path = os.path.join(subj_path, "t1_tensors", plane, f"{d}.pt")
                    if os.path.exists(tensor_path):
                        self.ids.append(d)
        
        elif self.structure == "flat":
            plane_dir = os.path.join(self.data_dir, plane)
            if os.path.exists(plane_dir):
                for f in os.listdir(plane_dir):
                    if f.endswith(".pt") and (not self.filter or self.filter in f):
                        self.ids.append(f.replace(".pt", ""))
                        
        self.ids.sort()
        print(f"[{plane.capitalize()}] Tensores validados: {len(self.ids)}")

    def __len__(self): 
        return len(self.ids)
        
    def __getitem__(self, idx):
        subj_id = self.ids[idx]
        if self.structure == "nested":
            file_path = os.path.join(self.data_dir, subj_id, "t1_tensors", self.plane, f"{subj_id}.pt")
        else:
            file_path = os.path.join(self.data_dir, self.plane, f"{subj_id}.pt")
            
        try:
            sample = torch.load(file_path, weights_only=False)
            return sample["image"].float(), float(sample["age"]), subj_id
        except Exception: 
            return None, None, subj_id

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([]), []
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch]

def detect_config(path):
    sd = torch.load(path, map_location="cpu")
    keys = sd.keys()
    indices = [int(k.split('.')[1]) for k in keys if "attnlist" in k and ".query" in k]
    nblock = max(indices) + 1 if indices else 6
    out_dim = sd['gloout.weight'].shape[0] if 'gloout.weight' in keys else 1
    return nblock, (out_dim > 1), sd

def get_predictions_TTA(model_key, config, dataset_cfg):
    USE_TTA = True if config['plane_type'] in ['axial', 'coronal'] else False
    model_path = os.path.join(MODELS_DIR, config['file'])
    
    if not os.path.exists(model_path): 
        return None, None

    nblock, is_dist, sd = detect_config(model_path)
    
    # Manejo de la excepción histórica del soft label
    if "model_sagittal_20260105-1206.pt" in model_path: is_dist = False
    
    model = GlobalLocalBrainAge(inplace=INPLACE, patch_size=PATCH_SIZE, step=STEP, 
                                nblock=nblock, backbone=config['backbone']).to(DEVICE)
    
    if is_dist:
        model.gloout = nn.Linear(model.gloout.in_features, 100).to(DEVICE)
        model.locout = nn.Linear(model.locout.in_features, 100).to(DEVICE)

    model.load_state_dict(sd)
    model.eval()

    ds = UnifiedInferenceDataset(dataset_cfg, config['plane_type'])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn_skip_none)

    preds, truths = {}, {}
    bins = torch.arange(100, device=DEVICE).float()

    with torch.no_grad():
        for images, ages, ids in tqdm(loader, desc=f"Inferencia {config['label']}", leave=False):
            if len(ids) == 0: continue
            images = images.to(DEVICE)
            
            def infer(img_tensor):
                out_logits = model(img_tensor)[0]
                if is_dist:
                    return (torch.softmax(out_logits, dim=1) * bins).sum(dim=1)
                return out_logits.flatten()

            out = infer(images)
            if USE_TTA:
                out = (out + infer(torch.flip(images, dims=[-1]))) / 2.0
                
            out = out.cpu().numpy()
            for i, subj_id in enumerate(ids):
                preds[subj_id] = out[i]
                truths[subj_id] = ages[i].item()
                
    return preds, truths

# ==============================================================================
# 4. MÉTRICAS BOOTSTRAP Y RENDERIZADO
# ==============================================================================
def calculate_bootstrap_metrics(y_true, y_pred, n_boot=1000, alpha=5.0):
    mae_l, r_l, r2_l, cs_l = [], [], [], []
    n = len(y_true)
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        
        abs_err = np.abs(yt - yp)
        mae_l.append(np.mean(abs_err))
        
        if np.std(yt) > 0 and np.std(yp) > 0:
            r_l.append(np.corrcoef(yt, yp)[0, 1])
        else: r_l.append(0)

        ss_tot = np.sum((yt - np.mean(yt))**2)
        r2_l.append(1 - (np.sum((yt - yp)**2) / ss_tot) if ss_tot > 0 else 0)
        cs_l.append(np.sum(abs_err <= alpha) / n * 100)
        
    return (np.mean(mae_l), np.std(mae_l), np.mean(r_l), np.std(r_l),
            np.mean(r2_l), np.std(r2_l), np.mean(cs_l), np.std(cs_l))

def plot_results(y_true, y_pred, db_name):
    sns.set_theme(style="white")
    fig, ax1 = plt.subplots(figsize=(10, 10))
    
    mae, r2, (pearson_r, _) = mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred), pearsonr(y_true, y_pred)
    point_errors = np.abs(y_pred - y_true)

    cmap_custom = sns.color_palette("blend:#6EE7B7,#053a2c", as_cmap=True)
    sc = ax1.scatter(y_true, y_pred, c=point_errors, cmap=cmap_custom, vmin=0, vmax=10, s=60, alpha=1.0, edgecolor='none')
                     
    min_v, max_v = min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5
    ax1.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, label="Identidad", alpha=0.5)
    
    # Ajuste dinámico de límites según la distribución del dataset
    ax1.set_xlim(min_v, max_v)
    ax1.set_ylim(min_v, max_v)
    ax1.margins(0)

    fmt_ax = ticker.FuncFormatter(lambda x, p: "" if x == 0 else (f'{int(x)}' if x % 1 == 0 else f'{x}'))
    ax1.xaxis.set_major_formatter(fmt_ax); ax1.yaxis.set_major_formatter(fmt_ax)

    for spine in ['top', 'right']: ax1.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']: 
        ax1.spines[spine].set_color('#d6d5d5')
        ax1.spines[spine].set_linewidth(2.0)
        
    ax1.tick_params(axis='both', color='#d6d5d5', labelcolor='black', width=2.0, length=5)
    ax1.set_title(f"Validación Externa OOD: {db_name}\n($R^2={r2:.3f}, r={pearson_r:.3f}, MAE={mae:.2f}$)", 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel("Edad Cronológica", fontsize=12); ax1.set_ylabel("Edad Estimada", fontsize=12)
    ax1.legend(fontsize=11)

    bg_patch = FancyBboxPatch((0.85, 0.10), width=0.10, height=0.30, boxstyle="round,pad=0.01",
                              fc="white", ec="0.8", alpha=0.95, zorder=50, transform=ax1.transAxes)
    ax1.add_patch(bg_patch)

    axins = inset_axes(ax1, width="3%", height="25%", loc='lower right', 
                       bbox_to_anchor=(-0.1, 0.11, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=axins, orientation="vertical", ticks=[0, 2.5, 5.0, 7.5, 10.0])
    cbar.outline.set_visible(False)
    for y in [0, 2.5, 5.0, 7.5, 10.0]: axins.hlines(y, 0, 1, colors='white', linewidths=1.5, zorder=100)
    axins.tick_params(axis='y', length=0, labelsize=11, pad=5)
    axins.set_title("MAE", fontsize=12, fontweight='bold', pad=10)

    out = os.path.join(PLOTS_DIR, f"inference_{db_name}_External_Validation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out

# ==============================================================================
# 5. ORQUESTADOR PRINCIPAL
# ==============================================================================
def run_inference_pipeline(db_name, db_cfg, stacker):
    print(f"\n" + "="*80)
    print(f">>> INICIANDO PIPELINE OOD: {db_name}")
    print("="*80)
    
    all_preds, gt, common = {}, None, None
    for k, cfg in BEST_MODELS.items():
        p, t = get_predictions_TTA(k, cfg, db_cfg)
        if not p: return
        all_preds[k] = p
        common = set(p.keys()) if common is None else common.intersection(set(p.keys()))
        gt = t

    ids = sorted(list(common))
    if not ids:
        print(f"[ERROR] No hay sujetos comunes en la intersección matricial para {db_name}.")
        return

    X = np.column_stack([[all_preds[k][i] for i in ids] for k in BEST_MODELS])
    y = np.array([gt[i] for i in ids])
    y_pred = stacker.predict(X)

    txt_buffer = []
    txt_buffer.append(f"{'Model':<16} | {'Age range':<12} | {'N':<4} | {'MAE (Mean ± SD)':<18} | {'Pearson r':<18} | {'CS (α=5)':<12}")
    txt_buffer.append("-" * 95)
    
    age_str, n_subj = f"{y.min():.1f} - {y.max():.1f}", len(y)
    metrics_summary = []

    for k in BEST_MODELS:
        pk = np.array([all_preds[k][i] for i in ids])
        mm, ms, rm, rs, _, _, cm, cs = calculate_bootstrap_metrics(y, pk)
        lbl = BEST_MODELS[k]['label']
        txt_buffer.append(f"{lbl:<16} | {age_str:<12} | {n_subj:<4} | {mm:.2f} ± {ms:.2f}".ljust(53) + f" | {rm:.3f} ± {rs:.3f}".ljust(21) + f" | {cm:.1f} ± {cs:.1f}%")

    txt_buffer.append("-" * 95)
    mm, ms, rm, rs, _, _, cm, cs = calculate_bootstrap_metrics(y, y_pred)
    txt_buffer.append(f"{'Ensemble ' + db_name[:6]:<16} | {age_str:<12} | {n_subj:<4} | {mm:.2f} ± {ms:.2f}".ljust(53) + f" | {rm:.3f} ± {rs:.3f}".ljust(21) + f" | {cm:.1f} ± {cs:.1f}%")
    
    metrics_summary.append({
        "Dataset": f"{db_name} (OOD)", "Age_Range": age_str, "N": n_subj, 
        "MAE_mean": mm, "MAE_sd": ms, "R_mean": rm, "R_sd": rs, "CS_mean": cm, "CS_sd": cs
    })

    print("\n".join(txt_buffer))
    
    csv_preds = os.path.join(BASE_DIR, f"inference_{db_name}_External_Predictions.csv")
    csv_metrics = os.path.join(BASE_DIR, f"inference_{db_name}_External_Metrics.csv")
    
    df_res = pd.DataFrame([
        {"Subject_ID": sid, "Chronological_Age": y[i], "Pred_Ensemble": y_pred[i], 
         **{f"Pred_{BEST_MODELS[k]['label'].replace(' ', '_')}": all_preds[k][sid] for k in BEST_MODELS}} 
        for i, sid in enumerate(ids)
    ])
    
    df_res.to_csv(csv_preds, index=False)
    pd.DataFrame(metrics_summary).to_csv(csv_metrics, index=False)
    
    plot_path = plot_results(y, y_pred, db_name)
    return plot_path, csv_preds, csv_metrics

def main():
    STACKER_FILE = os.path.join(MODELS_DIR, "best_ensemble_N=3_2.5045_20260326-1357.joblib") 
    if not os.path.exists(STACKER_FILE): sys.exit(f"Falta el Stacker: {STACKER_FILE}")

    stacker = joblib.load(STACKER_FILE)["model"]
    print(f"[*] Stacker Ridge CV cargado (Intercept: {stacker.intercept_:.4f})")

    targets = []
    if args.intecnus: targets.append("INTECNUS")
    if args.oasis: targets.append("OASIS3")
    if args.rrib: targets.append("RRIB")

    if not targets:
        print("Operación cancelada: Debés especificar al menos una cohorte (--intecnus, --oasis, --rrib, --all).")
        return

    out_plots, out_preds, out_metrics = [], [], []

    for t in targets:
        res = run_inference_pipeline(t, OOD_CONFIG[t], stacker)
        if res:
            plot_p, preds_p, metrics_p = res
            out_plots.append(plot_p)
            out_preds.append(preds_p)
            out_metrics.append(metrics_p)

    # --- REPORTE DE ARCHIVOS DE SALIDA ---
    print("\n" + "="*80)
    print(">>> RESUMEN DE ARCHIVOS EXPORTADOS")
    print("="*80)
    
    print("\n[1] Gráficos de Validación Multivariante (PNG):")
    for p in out_plots: 
        print(f"  -> {os.path.abspath(p)}")
    
    print("\n[2] Matrices de Predicción Vectorizada (CSV):")
    for p in out_preds: 
        print(f"  -> {os.path.abspath(p)}")
    
    print("\n[3] Reportes de Métricas Bootstrap (CSV):")
    for p in out_metrics: 
        print(f"  -> {os.path.abspath(p)}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()