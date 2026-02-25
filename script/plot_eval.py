import os

import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Konfigurasi Path ---
csv_name = "eval_20260223_051558.csv"
input_path = f"Skripsi/eval_logs/{csv_name}"
output_folder = "Skripsi/eval_logs/plot/"
output_path = os.path.join(output_folder, f"{csv_name.replace('.csv', '')}.png")

# Pastikan folder tujuan ada
os.makedirs(output_folder, exist_ok=True)

# --- 2. Membaca Data ---
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File {input_path} tidak ditemukan.")
    exit()

# --- 3. Visualisasi (4 Subplots) ---
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
plt.subplots_adjust(hspace=0.3)  # Memberi jarak antar grafik

# Subplot 1: Reward
axes[0].plot(df["step"], df["reward"], color="blue", linewidth=2)
axes[0].set_ylabel("Reward")
axes[0].set_title(f"Evaluation Metrics: {csv_name}", fontsize=14)
axes[0].grid(True, linestyle=":", alpha=0.7)

# Subplot 2: Distances (Dibedakan dari Reward)
axes[1].plot(df["step"], df["dist_ee_obj"], color="red", label="EE to Object")
axes[1].plot(df["step"], df["dist_obj_target"], color="green", label="Object to Target")
axes[1].set_ylabel("Distance (m)")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle=":", alpha=0.7)

# Subplot 3: Gripper State (Opening)
axes[2].plot(df["step"], df["gripper_opening"], color="purple", label="Opening Width")
axes[2].set_ylabel("Gripper Opening")
axes[2].grid(True, linestyle=":", alpha=0.7)

# Subplot 4: Gripper Joints (L & R)
axes[3].plot(df["step"], df["gripL_qpos"], color="orange", label="Left Joint")
axes[3].plot(
    df["step"], df["gripR_qpos"], color="brown", label="Right Joint", linestyle="--"
)
axes[3].set_xlabel("Step (Time)")
axes[3].set_ylabel("Joint Pos (rad)")
axes[3].legend(loc="upper right")
axes[3].grid(True, linestyle=":", alpha=0.7)

# --- 4. Simpan Hasil ---
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Selesai! Plot disimpan ke: {output_path}")

# Tampilkan jika perlu
plt.show()
