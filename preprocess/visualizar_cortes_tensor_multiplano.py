import torch
import nibabel as nib
import numpy as np
import os

# Ruta del archivo a inspeccionar
tensor_path = "../../data/processed/axial/sub-100053248969_preproc-quasiraw_T1w.pt"

# Cargar
data = torch.load(tensor_path)
image = data["image"].numpy()  # esperado: (5, 160, 160)

print("Forma original del tensor:", image.shape)

# Verificación
if image.shape != (5, 160, 160):
    raise ValueError(f"Forma inesperada: {image.shape}")

# Reordenar a (160, 160, 5) para formato NIfTI (x, y, z)
nii_data = np.transpose(image, (1, 2, 0))

# Crear y guardar NIfTI
nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4))
nib.save(nii_img, "visualizacion_cortes_axial.nii.gz")

print("NIfTI guardado.")
