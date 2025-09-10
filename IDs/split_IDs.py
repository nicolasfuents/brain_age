import random

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
INPUT_FILE = "train_ids_original.txt"  # copia del original (3227)
TRAIN_OUT = "train_ids.txt"            # sobrescribe con 3000
VALID_OUT = "valid_ids.txt"            # nuevo con 227
N_VALID = 227
SEED = 42

# -----------------------------
# CARGAR IDS Y HACER SPLIT
# -----------------------------
random.seed(SEED)
with open(INPUT_FILE, "r") as f:
    ids = [line.strip() for line in f.readlines()]

random.shuffle(ids)
valid_ids = ids[:N_VALID]
train_ids = ids[N_VALID:]

# -----------------------------
# GUARDAR ARCHIVOS
# -----------------------------
with open(TRAIN_OUT, "w") as f:
    f.write("\n".join(train_ids) + "\n")

with open(VALID_OUT, "w") as f:
    f.write("\n".join(valid_ids) + "\n")

print(f"Guardado:\n - {len(train_ids)} en train_ids.txt\n - {len(valid_ids)} en valid_ids.txt")
