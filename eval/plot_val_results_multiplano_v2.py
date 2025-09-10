# plot_val_results_multiplano_v2.py
# -----------------------------
# FIGURA: 4 modelos, cada uno con 2 filas
#  - Fila 1: scatter y_pred vs y_true + ideal + tendencia + banda 95%
#  - Fila 2: BAG (y_pred - y_true) vs edad real
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import os


# -----------------------------
# CONFIGURACIÓN Y RUTAS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLANES = ["axial", "coronal", "sagittal", "ensemble"]

DATA_DIR = os.path.join(BASE_DIR, "predictions")  # Carpeta donde guardes .txt
OUT_PATH = os.path.join(BASE_DIR, "scatter_val_all_models_v2.png")

fig = plt.figure(figsize=(14, 12))
main_gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)

for idx, plane in enumerate(PLANES):
    y_true = np.loadtxt(os.path.join(DATA_DIR, f"val_true_{plane}.txt"))
    y_pred = np.loadtxt(os.path.join(DATA_DIR, f"val_pred_{plane}.txt"))

    # Métricas (igual que antes)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    within_5 = np.sum(np.abs(y_true - y_pred) < 5)
    total = len(y_true)

    metrics_str = (
        f"MAE: {mae:.2f} años\n"
        f"MSE: {mse:.2f}\n"
        f"Pearson r: {r:.2f}\n"
        f"MAE < 5 años: {within_5}/{total} ({100*within_5/total:.1f}%)"
    )

    row, col = divmod(idx, 2)
    sub_gs = main_gs[row, col].subgridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.12)

    # --------- Fila 1: scatter + líneas + banda 95%
    sub_gs = main_gs[row, col].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.12)  # Adjusted height_ratios
    ax1 = fig.add_subplot(sub_gs[0])
    ax1.scatter(y_true, y_pred, alpha=0.7, label="Sujetos", color="steelblue")

    # Rango y líneas
    x_min, x_max = np.min(y_true), np.max(y_true)
    x_line = np.linspace(x_min, x_max, 200)

    # Ideal (misma apariencia)
    ax1.plot([x_min, x_max], [x_min, x_max], 'r--', label="Ideal (y = x)")

    # Tendencia (misma apariencia)
    m, b = np.polyfit(y_true, y_pred, 1)
    y_fit = m * x_line + b
    ax1.plot(x_line, y_fit, 'mediumpurple', alpha=0.6,
             label=f"Tendencia (y = {m:.2f}x + {b:.2f})")

    # Banda de confianza 95% del ajuste lineal (para la media)
    n = len(y_true)
    x_mean = np.mean(y_true)
    sxx = np.sum((y_true - x_mean) ** 2)
    resid = y_pred - (m * y_true + b)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2)) if n > 2 else 0.0
    se_line = s_err * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / sxx) if sxx > 0 else np.zeros_like(x_line)
    ci = 1.96 * se_line
    ax1.fill_between(x_line, y_fit - ci, y_fit + ci, color='mediumpurple', alpha=0.15, linewidth=0)

    ax1.set_xlabel("Edad real")
    ax1.set_ylabel("Edad predicha")
    ax1.set_title(f"Modelo {plane}")
    ax1.legend(framealpha=0.99, edgecolor='lightgray')
    ax1.grid(True)

    # Cuadro con métricas (idéntico a tu script)
    ax1.text(
        0.56, 0.05, metrics_str,  # Adjusted X-axis position from 0.65 to 0.60
        transform=ax1.transAxes,
        fontsize=9,
        va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='lightgray')
    )

    # --------- Fila 2: BAG vs edad real
    ax2 = fig.add_subplot(sub_gs[1], sharex=ax1)
    bag = y_pred - y_true
    ax2.scatter(y_true, bag, alpha=0.5, color="steelblue", s=18, label="BAG")
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel("BAG (pred − real)")
    ax2.set_xlabel("Edad real")
    ax2.grid(True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()
plt.close()

print("Visualización generada:", OUT_PATH)
