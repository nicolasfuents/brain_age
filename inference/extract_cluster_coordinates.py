"""
Dashboard Triplanar de Validación Neuroanatómica Híbrido (Atlas + HUD)
-----------------------------------------------------------------------------
- Integra segmentación exacta basada en atlas MNI (NIfTI) para estructuras profundas.
- Mantiene Bounding Boxes HUD para particiones corticales macroscópicas.
- Extrae contornos isométricos mediante marching squares para renderizado neón.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
from skimage.transform import resize as sk_resize

# ==============================================================================
# 1. CONFIGURACIÓN Y MAPEO DE ATLAS
# ==============================================================================
BASE_PROJECT = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"
DATA_DIR_OASIS = os.path.join(BASE_PROJECT, "data/OASIS3/MR_867_HC/processed")

# Resolución dinámica de ruta relativa al script (inference/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATLAS_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data/atlases/combined_subcortical_cerebellum_1mm.nii.gz"))

# IDs extraídos de combined_labels.csv
ATLAS_MAPPING = {
    "Ventriculos": [3, 14],
    "Ganglios": [4, 5, 6, 7, 15, 16, 17, 18],
    "Hipocampo": [9, 19],
    "Cerebelo": list(range(101, 150)) # Rango seguro para capturar todos los lobelillos > 100
}

# Cajas corticales (se mantienen como HUD)
CORTICAL_BOUNDS = {
    "sagittal": {"Prefrontal": {"y": (0.05, 0.45), "x": (0.65, 0.95)}, "Parietal": {"y": (0.05, 0.40), "x": (0.10, 0.60)}},
    "axial":    {"Frontal": {"y": (0.05, 0.40), "x": (0.20, 0.80)}, "Occipital": {"y": (0.65, 0.95), "x": (0.20, 0.80)}},
    "coronal":  {"Cortical_Sup": {"y": (0.05, 0.35), "x": (0.10, 0.90)}}
}

STYLE = {
    "facecolor": "black",
    "textcolor": "white",
    "roi_colors": {
        "Prefrontal": "#39FF14", "Parietal": "#FE019A", "Cerebelo": "#FF9900",
        "Frontal": "#39FF14", "Ganglios": "#FF9900", "Occipital": "#FE019A",
        "Cortical_Sup": "#00FFCC", "Ventriculos": "#FFD700", "Hipocampo": "#FE019A"
    },
    "dot_colors": {"Youth_Blue": "#00FFFF", "Aging_Red": "#FF003C"}
}

def extract_clusters(cam, threshold_pct=85, min_size=5):
    results = []
    for sign, label in [(-1, "Youth_Blue"), (1, "Aging_Red")]:
        data = np.where(np.sign(cam) == sign, np.abs(cam), 0)
        if not np.any(data > 0): continue
        
        thresh = np.percentile(data[data > 0], threshold_pct)
        binary = (data >= thresh).astype(int)
        labels = measure.label(binary)
        props = measure.regionprops(labels, intensity_image=data)
        
        for p in props:
            if p.area < min_size: continue
            y, x = p.weighted_centroid
            results.append({"Sign": label, "Centroid_Y": y, "Centroid_X": x, "Mass": p.mean_intensity * p.area, "Max": p.max_intensity})
    return results

def main():
    input_dir = "saliency_maps_outliers"
    output_dir = "resultados_robustez"
    viz_dir = os.path.join(output_dir, "validacion_triplanar_atlas")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Carga estática del volumen del atlas
    if not os.path.exists(ATLAS_PATH):
        sys.exit(f"[-] ERROR: Atlas no encontrado en {ATLAS_PATH}")
    atlas_vol = nib.load(ATLAS_PATH).get_fdata()
    dim_x, dim_y, dim_z = atlas_vol.shape
    
    files = [f for f in os.listdir(input_dir) if f.startswith("gradcam_raw_") and f.endswith(".npy")]
    subjects = sorted(list(set(["_".join(f.replace("gradcam_raw_", "").replace(".npy", "").split("_")[:-1]) for f in files])))

    if not subjects:
        sys.exit("[-] ERROR: No se encontraron archivos para procesar.")

    print(f"[*] Procesando {len(subjects)} sujetos con renderizado topográfico de contornos...")

    all_subj_data = [] # Se inicializa afuera para acumular la métrica de toda la cohorte

    for subj_id in subjects:
        fig_triplanar, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=STYLE['facecolor'])
        axes = axes.flatten()
        for ax in axes: ax.set_axis_off()
        plt.suptitle(f"Dashboard Híbrido de Validación (Atlas MNI) - ID: {subj_id}", color=STYLE['textcolor'], fontsize=18, fontweight='bold', y=1.05)
        
        panel_map = {"axial": 0, "coronal": 1, "sagittal": 2}

        for plane in ["axial", "coronal", "sagittal"]:
            file = f"gradcam_raw_{subj_id}_{plane}.npy"
            if not os.path.exists(os.path.join(input_dir, file)): continue
            
            cam_raw = np.load(os.path.join(input_dir, file))
            
            tensor_path = os.path.join(DATA_DIR_OASIS, subj_id, "t1_tensors", plane, f"{subj_id}.pt")
            if not os.path.exists(tensor_path): continue
                
            sample = torch.load(tensor_path, weights_only=False, map_location="cpu")
            img_slice = sample["image"][2].float().numpy()
            
            # --- ROTACIÓN GEOMÉTRICA (90° IZQUIERDA PARA TODOS) ---
            if plane == "sagittal":
                cam_rot = np.rot90(cam_raw, k=1)
                img_display_base = np.rot90(img_slice, k=1)
                atlas_slice = np.rot90(atlas_vol[dim_x//2, :, :], k=1)
            elif plane == "coronal":
                cam_rot = np.rot90(cam_raw, k=2) # El Grad-CAM requiere k=2 para preservar la alineación con T1 (k=1)
                img_display_base = np.rot90(img_slice, k=1)
                atlas_slice = np.rot90(atlas_vol[:, dim_y//2, :], k=1)
            else: # axial
                cam_rot = np.rot90(cam_raw, k=2) 
                img_display_base = np.rot90(img_slice, k=1)
                atlas_slice = np.rot90(atlas_vol[:, :, dim_z//2], k=1)
            
            img_norm = (img_display_base - img_display_base.min()) / (img_display_base.max() - img_display_base.min() + 1e-8)
            H, W = img_norm.shape
            
            cam_high = sk_resize(cam_rot, (H, W), order=1, preserve_range=True)
            mask = img_norm > 0.05
            cam_anatomical = np.where(mask, cam_high, 0.0)
            clusters = extract_clusters(cam_anatomical)
            cam_display = np.where(mask, cam_high, np.nan) 
            
            # --- CORTE DEL ATLAS SEGÚN PLANO ---
            
            atlas_slice_resized = sk_resize(atlas_slice, (H, W), order=0, preserve_range=True)
            
            ax = axes[panel_map[plane]]
            ax.set_title(plane.upper(), color=STYLE['textcolor'], fontsize=14, fontweight='bold', pad=10)
            ax.imshow(img_norm, cmap='gray', interpolation='bicubic')
            ax.imshow(cam_display, cmap='seismic', vmin=-1, vmax=1, alpha=0.55, interpolation='bilinear')
            
            # Renderizado de Centroides y Asignación Híbrida de Zona
            for c in clusters:
                y_pixel, x_pixel = c["Centroid_Y"], c["Centroid_X"]
                
                zone_assigned = "Otras_Cortex"
                
                # 1. Prioridad Atlas: Verificar si el centroide cae en una estructura profunda
                val_atlas = atlas_slice_resized[int(y_pixel), int(x_pixel)]
                for region_name, labels in ATLAS_MAPPING.items():
                    if val_atlas in labels:
                        zone_assigned = region_name
                        break
                        
                # 2. Prioridad HUD: Si no es profunda, verificar si cae en una caja cortical
                if zone_assigned == "Otras_Cortex":
                    y_n, x_n = y_pixel / H, x_pixel / W
                    for z_name, bounds in CORTICAL_BOUNDS.get(plane, {}).items():
                        if bounds["y"][0] <= y_n <= bounds["y"][1] and bounds["x"][0] <= x_n <= bounds["x"][1]:
                            zone_assigned = z_name
                            break
                
                c.update({"Subject_ID": subj_id, "Plane": plane, "Zone": zone_assigned})
                all_subj_data.append(c)
                
                # Renderizado Neón
                color = STYLE['dot_colors'][c["Sign"]]
                ax.scatter(x_pixel, y_pixel, c=color, s=400, alpha=0.15, edgecolors='none', zorder=4)
                ax.scatter(x_pixel, y_pixel, facecolors='none', edgecolors=color, s=120, linewidth=1.5, alpha=0.9, zorder=5)
                ax.scatter(x_pixel, y_pixel, c='white', s=25, edgecolors=color, linewidth=1.2, zorder=6)

            # Renderizado de Contornos Anatómicos (Estructuras Profundas)
            for region_name, labels in ATLAS_MAPPING.items():
                # Solo renderizar si el atlas tiene presencia en este corte
                region_mask = np.isin(atlas_slice_resized, labels)
                if np.any(region_mask):
                    contours = measure.find_contours(region_mask, 0.5)
                    color = STYLE['roi_colors'].get(region_name, "#FFFFFF")
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=4, alpha=0.3, zorder=3) # Glow
                        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5, zorder=4) # Core
                    
                    # Etiqueta en el centro de masa de la región
                    y_idx, x_idx = np.where(region_mask)
                    ax.text(np.mean(x_idx), np.mean(y_idx), f"[{region_name.upper()}]", color=color, 
                            fontsize=8, fontweight='bold', fontfamily='monospace', ha='center',
                            bbox=dict(facecolor='black', edgecolor=color, alpha=0.75, boxstyle='round,pad=0.2'), zorder=5)

            # Renderizado de Bounding Boxes (Córtex)
            cortical = CORTICAL_BOUNDS.get(plane, {})
            for zone, b in cortical.items():
                x0, y0 = b['x'][0] * W, b['y'][0] * H
                width, height = (b['x'][1] - b['x'][0]) * W, (b['y'][1] - b['y'][0]) * H
                color = STYLE['roi_colors'][zone]
                
                glow = patches.FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0,rounding_size=12", linewidth=6, edgecolor=color, facecolor='none', alpha=0.2, zorder=3)
                ax.add_patch(glow)
                rect = patches.FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0,rounding_size=12", linewidth=2, edgecolor=color, facecolor='none', alpha=0.9, zorder=4)
                ax.add_patch(rect)
                ax.text(x0 + 8, y0 + 24, f"[{zone.upper()}]", color=color, fontsize=8, fontweight='bold', fontfamily='monospace', bbox=dict(facecolor='black', edgecolor=color, alpha=0.75, boxstyle='round,pad=0.2'), zorder=5)

        out_img = os.path.join(viz_dir, f"atlas_hud_{subj_id}.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=fig_triplanar.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"  [+] Generado: {os.path.abspath(out_img)}")

    df = pd.DataFrame(all_subj_data)
    csv_path = os.path.join(output_dir, "cuantificacion_triplanar_phd.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*85)
    print(f"{'SUBJECT ID':<20} | {'PLANE':<10} | {'ZONE':<12} | {'MASS':<10} | {'MAX INT'}")
    print("-" * 85)
    
    # Volcado completo de la tabla iterando sobre la cohorte entera, ordenada lógicamente
    df_sorted = df.sort_values(by=["Subject_ID", "Mass"], ascending=[True, False])
    for _, row in df_sorted.iterrows():
        print(f"{row['Subject_ID']:<20} | {row['Plane']:<10} | {row['Zone']:<12} | {row['Mass']:<10.2f} | {row['Max']: .4f}")

if __name__ == "__main__":
    main()