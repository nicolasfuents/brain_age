# Usage: python inference_oasis.py
import os
import sys
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from tqdm import tqdm

# Importar arquitectura
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from GlobalLocalTransformer_soft_labels import GlobalLocalBrainAge
except ImportError:
    sys.exit("CRITICAL: No se pudo importar GlobalLocalTransformer.")

# ==============================================================================
# 1. CONFIGURACION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path

MODELS_DIR = os.path.join(BASE_DIR, "../models")
PLOTS_DIR = os.path.join(BASE_DIR, "../plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Modelos anteriores (Mayo)
MAYO_MODELS = {
    "axi_soft": {
        "file": "BATCH20_MAYO_556264/best_model_axial_axial_resnet18_soft_n8.pt", 
        "backbone": "resnet18", 
        "plane_type": "axial", 
        "label": "Axial R18 Soft"
    },
    "cor_smoothl1": {
        "file": "BATCH11_MAYO_556254/best_model_coronal_coronal_resnet34_smoothl1_n8.pt", 
        "backbone": "resnet34", 
        "plane_type": "coronal", 
        "label": "Coronal R34 SL1"
    },
    "sag_smoothl1": {
        "file": "BATCH8_MAYO_556251/best_model_sagittal_sagittal_resnet18_smoothl1_n6.pt", 
        "backbone": "resnet18", 
        "plane_type": "sagittal", 
        "label": "Sagittal R18 SL1"
    }
}

# Modelos actuales (Best_3_Triplanar_Global - Junio ipw)
IPW_MODELS = {
    "axi_soft": {
        "file": "BATCH19_JUNIO_ipw_583181/best_model_axial_axial_resnet18_soft_n6.pt", 
        "backbone": "resnet18", 
        "plane_type": "axial", 
        "label": "Axial R18 Soft"
    },
    "cor_smoothl1": {
        "file": "BATCH11_JUNIO_ipw_583173/best_model_coronal_coronal_resnet34_smoothl1_n8.pt", 
        "backbone": "resnet34", 
        "plane_type": "coronal", 
        "label": "Coronal R34 SL1"
    },
    "sag_mse": {
        "file": "BATCH19_JUNIO_ipw_583181/best_model_sagittal_sagittal_resnet18_mse_n6.pt", 
        "backbone": "resnet18", 
        "plane_type": "sagittal", 
        "label": "Sagittal R18 MSE"
    }
}

# Modelos actuales (Best_3_Triplanar_Global - Junio ipw v2)
IPW_MODELS_v2 = {
    "axi_soft": {
        "file": "BATCH1_JUNIO_ipw_opt_624618/best_model_axial_axial_resnet18_soft_n6_ipw.pt", 
        "backbone": "resnet18", 
        "plane_type": "axial", 
        "label": "Axial R18 Soft"
    },
    "cor_smoothl1": {
        "file": "BATCH1_JUNIO_ipw_opt_624618/best_model_coronal_coronal_resnet34_smoothl1_n8_ipw.pt", 
        "backbone": "resnet34", 
        "plane_type": "coronal", 
        "label": "Coronal R34 SL1"
    },
    "sag_smoothl1": {
        "file": "BATCH1_JUNIO_ipw_opt_624618/best_model_sagittal_sagittal_resnet18_smoothl1_n6_ipw.pt", 
        "backbone": "resnet18", 
        "plane_type": "sagittal", 
        "label": "Sagittal R18 SL1"
    }
}

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PROJECT = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"

# NUEVA RUTA APUNTADA AL DOMINIO ARMONIZADO (P99)
DATA_DIR_OASIS = os.path.join(BASE_PROJECT, "data/HARMONIZED_OOD_P99/MR_867_HC")

PATCH_SIZE = 64; STEP = 32; NBLOCK = 8; INPLACE = 5

# ==============================================================================
# 2. LOGICA DE DATOS ADAPTADA A CARPETAS
# ==============================================================================
class InferenceDataset(Dataset):
    def __init__(self, data_dir, plane):
        self.plane = plane
        self.data_dir = data_dir
        self.ids = []
        self.file_map = {}
        
        p_dir = Path(data_dir)
        if p_dir.exists():
            for p in p_dir.rglob(f"**/{plane}/*.pt"):
                if "OAS" not in p.stem: continue
                self.ids.append(p.stem)
                self.file_map[p.stem] = str(p)
                
        self.ids.sort()
        print(f"[{plane.capitalize()}] Sujetos detectados (Dominio Armonizado): {len(self.ids)}")

    def __len__(self): 
        return len(self.ids)
        
    def __getitem__(self, idx):
        subj_id = self.ids[idx]
        file_path = self.file_map[subj_id]
        try:
            sample = torch.load(file_path, weights_only=False)
            return sample["image"].float(), float(sample["age"]), subj_id
        except Exception as e: 
            print(f"Error cargando {subj_id}: {e}")
            return None, None, subj_id

def detect_config(path):
    try:
        sd = torch.load(path, map_location="cpu")
        keys = sd.keys()
        indices = [int(k.split('.')[1]) for k in keys if "attnlist" in k and ".query" in k]
        nblock = max(indices) + 1 if indices else 6
        out_dim = sd['gloout.weight'].shape[0] if 'gloout.weight' in keys else 1
        is_dist = (out_dim > 1)
        return nblock, is_dist, sd
    except Exception as e:
        print(f"Error detectando config: {e}")
        return 6, False, None
    
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([]), []
    images = torch.stack([b[0] for b in batch])
    ages = torch.tensor([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return images, ages, ids

def get_predictions_TTA(model_key, config):
    USE_TTA = True if config['plane_type'] in ['axial', 'coronal'] else False
    status = "ON" if USE_TTA else "OFF"
    
    model_path = os.path.join(MODELS_DIR, config['file'])
    if not os.path.exists(model_path): 
        print(f"No encontrado: {model_path}")
        return None, None

    nblock_detected, is_dist, sd = detect_config(model_path)
    if sd is None: return None, None
    
    print(f"\n- {config['label']}: TTA {status} | Blocks: {nblock_detected} | SoftLabels: {is_dist}")

    model = GlobalLocalBrainAge(inplace=INPLACE, patch_size=PATCH_SIZE, step=STEP, 
                                nblock=nblock_detected, backbone=config['backbone']).to(DEVICE)
    
    if is_dist:
        model.gloout = torch.nn.Linear(model.gloout.in_features, 100).to(DEVICE)
        model.locout = torch.nn.Linear(model.locout.in_features, 100).to(DEVICE)

    model.load_state_dict(sd)
    model.eval()

    ds = InferenceDataset(DATA_DIR_OASIS, config['plane_type'])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn_skip_none)

    preds, truths = {}, {}
    bins = torch.arange(100, device=DEVICE).float()

    with torch.no_grad():
        for images, ages, ids in tqdm(loader, desc="Inferencia", leave=False):
            if len(ids) == 0: continue
            images = images.to(DEVICE)
            
            def infer(img_tensor):
                outputs = model(img_tensor)
                batch_preds = []
                for out_head in outputs:
                    if is_dist:
                        probs = torch.softmax(out_head, dim=1)
                        batch_preds.append((probs * bins).sum(dim=1))
                    else:
                        batch_preds.append(out_head.flatten())
                # Promediamos la predicción de la rama global y todas las locales
                return torch.stack(batch_preds).mean(dim=0)

            out = infer(images)
            
            if USE_TTA:
                out_flip = infer(torch.flip(images, dims=[-1]))
                out = (out + out_flip) / 2.0
                
            out = out.cpu().numpy()
            for i, subj_id in enumerate(ids):
                preds[subj_id] = out[i]
                truths[subj_id] = ages[i].item()
                
    return preds, truths

def calculate_bootstrap_metrics(y_true, y_pred, n_boot=1000, alpha=5.0):
    # 1. Estimador puntual empírico estricto sobre N original
    abs_err_exact = np.abs(y_true - y_pred)
    exact_mae = np.mean(abs_err_exact)
    
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        exact_r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        exact_r = 0.0

    ss_res_exact = np.sum((y_true - y_pred)**2)
    ss_tot_exact = np.sum((y_true - np.mean(y_true))**2)
    exact_r2 = 1 - (ss_res_exact / ss_tot_exact) if ss_tot_exact > 0 else 0.0
    
    exact_cs = (np.sum(abs_err_exact <= alpha) / len(y_true)) * 100.0

    # 2. Estimación de incertidumbre vía Bootstrap
    mae_list, r_list, r2_list, cs_list = [], [], [], []
    n = len(y_true)
    
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]
        
        abs_err = np.abs(yt_sample - yp_sample)
        mae_list.append(np.mean(abs_err))
        
        if np.std(yt_sample) > 0 and np.std(yp_sample) > 0:
            r_list.append(np.corrcoef(yt_sample, yp_sample)[0, 1])
        else:
            r_list.append(0.0)

        ss_res_boot = np.sum((yt_sample - yp_sample)**2)
        ss_tot_boot = np.sum((yt_sample - np.mean(yt_sample))**2)
        r2_list.append(1 - (ss_res_boot / ss_tot_boot) if ss_tot_boot > 0 else 0.0)
            
        cs_list.append(np.sum(abs_err <= alpha) / n * 100.0)
        
    # Retornamos el valor puntual exacto junto a la desviación estándar del ensamble de remuestreo
    return (exact_mae, np.std(mae_list), 
            exact_r, np.std(r_list),
            exact_r2, np.std(r2_list),
            exact_cs, np.std(cs_list))

# ==============================================================================
# 3. METRICAS Y PLOTTEO
# ==============================================================================
def calculate_bootstrap_metrics(y_true, y_pred, n_boot=1000, alpha=5.0):
    mae_list, r_list, r2_list, cs_list = [], [], [], []
    n = len(y_true)
    
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]
        
        abs_err = np.abs(yt_sample - yp_sample)
        mae_list.append(np.mean(abs_err))
        
        if np.std(yt_sample) > 0 and np.std(yp_sample) > 0:
            r_val = np.corrcoef(yt_sample, yp_sample)[0, 1]
            r_list.append(r_val)
        else:
            r_list.append(0)

        ss_res = np.sum((yt_sample - yp_sample)**2)
        ss_tot = np.sum((yt_sample - np.mean(yt_sample))**2)
        if ss_tot > 0: 
            r2_list.append(1 - (ss_res / ss_tot))
        else: 
            r2_list.append(0)
            
        cs_list.append(np.sum(abs_err <= alpha) / n * 100)
        
    return (np.mean(mae_list), np.std(mae_list), 
            np.mean(r_list), np.std(r_list),
            np.mean(r2_list), np.std(r2_list),
            np.mean(cs_list), np.std(cs_list))

def plot_results(y_true, y_pred, labels, suffix=""):
    print("\nGenerando gráficos finales...")
    sns.set_theme(style="white")
    
    # --------------------------------------------------------------------------
    # FIGURA 1: Regresión Lineal y Análisis de BAG (Brain Age Gap)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    residuals = y_pred - y_true  # BAG
    me = np.mean(residuals)
    point_errors = np.abs(residuals)
    
    # Auditoría del sesgo etario (Age Bias)
    r_bag, p_bag = pearsonr(y_true, residuals)

    # Paleta Cyber-Emerald (ajustada sin glow)
    cmap_custom = sns.color_palette("blend:#6EE7B7,#053a2c", as_cmap=True)
    
    # --- SUBPLOT 1: Edad Cronológica vs Edad Estimada ---
    sc = ax1.scatter(y_true, y_pred, c=point_errors, cmap=cmap_custom, 
                     vmin=0, vmax=10, s=100, alpha=1.0, edgecolor='none')
                     
    min_v, max_v = min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5
    ax1.plot([min_v, max_v], [min_v, max_v], "k--", lw=2.0, label="Identity", alpha=0.5)
    
    ax1.set_xlim(30, 100)
    ax1.set_ylim(30, 100)
    ax1.margins(0)

    # Formateadores de ejes
    fmt_ax_hide_zero = ticker.FuncFormatter(lambda x, p: "" if x == 0 else (f'{int(x)}' if x % 1 == 0 else f'{x}'))
    fmt_ax_standard = ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x % 1 == 0 else f'{x}')
    
    ax1.xaxis.set_major_formatter(fmt_ax_hide_zero)
    ax1.yaxis.set_major_formatter(fmt_ax_hide_zero)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('#d6d5d5')
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('#d6d5d5')
    ax1.spines['bottom'].set_linewidth(2.0)
    ax1.tick_params(axis='both', color='#d6d5d5', labelcolor='black', width=2.0, length=6, labelsize=14)

    ax1.annotate("0", xy=(0, 0), xycoords='data', xytext=(-7, -8), textcoords='offset points', ha='right', va='top')
                 
    ax1.set_title(f"External Validation on OASIS-3\n($R^2={r2:.3f}, r={pearson_r:.3f}, MAE={mae:.2f}, ME={me:.2f}$)", 
                  fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Chronological Age (Years)", fontsize=15)
    ax1.set_ylabel("Estimated Age (Years)", fontsize=15)
    ax1.legend(loc='upper left', fontsize=13)

    bg_patch1 = FancyBboxPatch((0.84, 0.08), width=0.12, height=0.34,
                               boxstyle="round,pad=0.01", fc="white", ec="0.8", alpha=0.95, zorder=50,
                               transform=ax1.transAxes)
    ax1.add_patch(bg_patch1)

    axins = inset_axes(ax1, width="3%", height="25%", loc='lower right', 
                       bbox_to_anchor=(-0.1, 0.11, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
    
    cbar = fig.colorbar(sc, cax=axins, orientation="vertical", ticks=[0, 2.5, 5.0, 7.5, 10.0])
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x >= 10 else f'{x}'))
    cbar.outline.set_visible(False)
    
    for y in [0, 2.5, 5.0, 7.5, 10.0]:
        axins.hlines(y, xmin=0, xmax=1, colors='white', linewidths=1.5, zorder=100)
    
    axins.tick_params(axis='y', length=0, labelsize=13, pad=5)
    axins.set_title("MAE", fontsize=14, fontweight='bold', pad=11, x=0.9)

    # --- SUBPLOT 2: Brain Age Gap vs Edad Cronológica ---
    ax2.scatter(y_true, residuals, c=point_errors, cmap=cmap_custom, 
                vmin=0, vmax=10, s=100, alpha=1.0, edgecolor='none', label="Subjects")
    
    ax2.axhline(0, color='k', linestyle='--', lw=2.0, alpha=0.5, label="Ref. (BAG=0)")
    
    # Lógica de línea de tendencia para BAG
    z_bag = np.polyfit(y_true, residuals, 1)
    a_bag, b_bag = z_bag
    p_poly_bag = np.poly1d(z_bag)
    trend_x_bag = np.array([30, 100])
    trend_y_bag = p_poly_bag(trend_x_bag)
    
    # Plotteo de la línea de tendencia con fórmula en el legend
    signo = "+" if b_bag >= 0 else "-"
    label_trend = f"Trend: {a_bag:.3f}x {signo} {abs(b_bag):.3f}"
    ax2.plot(trend_x_bag, trend_y_bag, color='k', linestyle='-', lw=2.0, alpha=0.8, label=label_trend)
    
    ax2.set_xlim(35, 100)
    ax2.set_ylim(-50, 50)

    ax2.xaxis.set_major_formatter(fmt_ax_hide_zero)
    ax2.yaxis.set_major_formatter(fmt_ax_standard)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_color('#d6d5d5')
    ax2.spines['left'].set_linewidth(2.0)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('#d6d5d5')
    ax2.spines['bottom'].set_linewidth(2.0)
    ax2.tick_params(axis='both', color='#d6d5d5', labelcolor='black', width=2.0, length=6, labelsize=14)
    
    ax2.set_title(f"Brain Age Gap (BAG) vs. Chronological Age\n(Age Bias: $r={r_bag:.3f}, p={p_bag:.3e}$)", 
                  fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel("Chronological Age (Years)", fontsize=15)
    ax2.set_ylabel("BAG (Years)", fontsize=15)
    ax2.legend(fontsize=13)

    out_main = os.path.join(PLOTS_DIR, f"inference_OASIS3_External_Validation_and_BAG{suffix}.png")
    plt.tight_layout()
    plt.savefig(out_main, dpi=300, bbox_inches='tight')
    print(f"[*] Gráfico principal guardado en: {out_main}")

    # --------------------------------------------------------------------------
    # FIGURA 2: Distribución Etaria (KDE + Histograma Moderno)
    # --------------------------------------------------------------------------
    fig_dist, ax3 = plt.subplots(figsize=(9, 7))
    
    # Utilizando los colores primarios de OASIS para mantener coherencia
    sns.histplot(y_true, bins=15, kde=True, color="#6EE7B7", edgecolor="none", alpha=0.5, line_kws={'color': '#053a2c', 'lw': 2.5}, ax=ax3)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(True)
    ax3.spines['left'].set_color('#d6d5d5')
    ax3.spines['left'].set_linewidth(2.0)
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['bottom'].set_color('#d6d5d5')
    ax3.spines['bottom'].set_linewidth(2.0)
    ax3.tick_params(axis='both', color='#d6d5d5', labelcolor='black', width=2.0, length=6, labelsize=14)
    
    ax3.xaxis.set_major_formatter(fmt_ax_hide_zero)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax3.set_xlim(30, 100)
    
    ax3.set_title(f"Age Distribution (OASIS-3, N={len(y_true)})", fontsize=18, fontweight='bold', pad=15)
    ax3.set_xlabel("Chronological Age (Years)", fontsize=15)
    ax3.set_ylabel("Frequency", fontsize=15)
    
    out_dist = os.path.join(PLOTS_DIR, f"inference_OASIS3_External_AgeDistribution{suffix}.png")
    fig_dist.tight_layout()
    fig_dist.savefig(out_dist, dpi=300, bbox_inches='tight')
    print(f"[*] Gráfico de distribución etaria guardado en: {out_dist}")

def main():
    runs = [
        {
            "suffix": "",
            "models": MAYO_MODELS,
            "stacker": os.path.join(MODELS_DIR, "ensemble_results", "Best_3_Triplanar_Global_GLOBAL_96_MODELS_preproc_corregido_y_ADNI_saneado_ridge.joblib")
        },
        {
            "suffix": "_ipw",
            "models": IPW_MODELS,
            "stacker": os.path.join(MODELS_DIR, "ensemble_results", "Best_3_Triplanar_Global_GLOBAL_96_MODELS_JUNIO_ipw_ridge.joblib")
        },
        {
            "suffix": "_ipw_v2",
            "models": IPW_MODELS_v2,
            "stacker": os.path.join(MODELS_DIR, "ensemble_results", "Best_3_Triplanar_Global_BATCH1_JUNIO_ipw_opt_624618_ridge.joblib")
        }
    ]
    
    for run in runs:
        suffix = run["suffix"]
        models_config = run["models"]
        stacker_path = run["stacker"]
        
        print(f"\n==================================================================================")
        print(f"INICIANDO EVALUACIÓN EN OASIS-3 - CONFIGURACIÓN: {suffix.upper() if suffix else 'MAYO (UNWEIGHTED)'}")
        print(f"==================================================================================")
        
        if not os.path.exists(stacker_path):
            print(f"WARNING: No se encontró el archivo de stacker {stacker_path}. Saltando...")
            continue

        artifact = joblib.load(stacker_path)
        stacker = artifact["model"] if isinstance(artifact, dict) else artifact
        
        print(f"Stacker cargado correctamente. Intercept: {stacker.intercept_:.4f}")

        all_preds, gt, common = {}, None, None
        
        for k, cfg in models_config.items():
            p, t = get_predictions_TTA(k, cfg)
            if p is None:
                print(f"ERROR: Fallo al obtener predicciones para {k}.")
                return
            all_preds[k] = p
            if common is None: 
                common = set(p.keys()); gt = t
            else: 
                common = common.intersection(set(p.keys()))

        ids = sorted(list(common))
        print(f"\nSujetos con predicciones completas en todos los modelos alineados: {len(ids)}")
        
        if len(ids) == 0:
            print("No hay sujetos suficientes para graficar.")
            continue

        X = np.column_stack([[all_preds[k][i] for i in ids] for k in models_config])
        y = np.array([gt[i] for i in ids])
        y_pred = stacker.predict(X)
        
        # --- EVALUACIÓN CON BOOTSTRAP (TABLA TIPO LATEX) ---
        print("\n" + "="*95)
        print(f"--- EVALUACIÓN EXTERNA EN OASIS3 ({suffix.upper() if suffix else 'MAYO'}) CON BOOTSTRAP (N=1000) ---")
        
        txt_buffer = []
        header = f"{'Model':<16} | {'Age range':<12} | {'N':<4} | {'MAE (Mean ± SD)':<18} | {'Pearson r':<18} | {'CS (α=5)':<12}"
        txt_buffer.append(header)
        txt_buffer.append("-" * 95)
        
        age_range_str = f"{y.min():.1f} - {y.max():.1f}"
        n_subj = len(y)
        metrics_summary = []

        # Modelos Base
        for k in models_config:
            preds_k = np.array([all_preds[k][i] for i in ids])
            mae_m, mae_s, r_m, r_s, r2_m, r2_s, cs_m, cs_s = calculate_bootstrap_metrics(y, preds_k)
            
            lbl = models_config[k]['label']
            line_str = (f"{lbl:<16} | {age_range_str:<12} | {n_subj:<4} | "
                        f"{mae_m:.2f} ± {mae_s:.2f}".ljust(18) + " | " + 
                        f"{r_m:.3f} ± {r_s:.3f}".ljust(18) + " | " + 
                        f"{cs_m:.1f} ± {cs_s:.1f}%")
            txt_buffer.append(line_str)

        txt_buffer.append("-" * 95)

        # Ensamble Final
        mae_m, mae_s, r_m, r_s, r2_m, r2_s, cs_m, cs_s = calculate_bootstrap_metrics(y, y_pred)
        line_str = (f"{'Ensemble OASIS3':<16} | {age_range_str:<12} | {n_subj:<4} | "
                    f"{mae_m:.2f} ± {mae_s:.2f}".ljust(18) + " | " + 
                    f"{r_m:.3f} ± {r_s:.3f}".ljust(18) + " | " + 
                    f"{cs_m:.1f} ± {cs_s:.1f}%")
        txt_buffer.append(line_str)
        metrics_summary.append({
            "Dataset": "OASIS3 (External)", "Age_Range": age_range_str, "N": n_subj, 
            "MAE_mean": mae_m, "MAE_sd": mae_s, "R_mean": r_m, "R_sd": r_s, "CS_mean": cs_m, "CS_sd": cs_s
        })
        
        for line in txt_buffer:
            print(line)
        print("="*95 + "\n")
        
        # --- EXPORTACIÓN ---
        csv_preds_path = os.path.join(BASE_DIR, f"inference_OASIS3_External_Predictions{suffix}.csv")
        csv_metrics_path = os.path.join(BASE_DIR, f"inference_OASIS3_External_Metrics{suffix}.csv")
        
        results_list = []
        for idx, subj_id in enumerate(ids):
            row_data = {"Subject_ID": subj_id, "Chronological_Age": y[idx], "Pred_Ensemble": y_pred[idx]}
            for k in models_config:
                row_data[f"Pred_{models_config[k]['label'].replace(' ', '_')}"] = all_preds[k][subj_id]
            results_list.append(row_data)
            
            pd.DataFrame(results_list).to_csv(csv_preds_path, index=False)
            pd.DataFrame(metrics_summary).to_csv(csv_metrics_path, index=False)
        print(f"Datos guardados en CSV para futura concatenación con la tabla multicéntrica.")

        plot_results(y, y_pred, [models_config[k]['label'] for k in models_config], suffix=suffix)

if __name__ == "__main__":
    main()
