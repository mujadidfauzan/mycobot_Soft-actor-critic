import time

import mujoco
import mujoco.viewer
import numpy as np

# --- KONFIGURASI XML ---
XML_PATH = "/home/fauzan/Mujoco/Skripsi/source/robot/robot2.xml"

# --- DEFINISI SUDUT SENDI (Radian) ---
# Format: [J1, J2, J3, J4, J5, J6]
# J1: Base (Kiri-Kanan)
# J2: Lengan Bawah (Maju-Mundur) -> Negatif = Maju
# J3: Siku (Naik-Turun)
# J4: Rotasi Lengan
# J5: Pergelangan (Angguk) -> 1.57 (90 derajat) biar tegak lurus lantai
# J6: Putaran Ujung

# 1. Posisi Home (Tegak)
POSE_HOME = [0, 0, 0, 0, 0, 0]

# 2. Posisi Siap (Membungkuk sedikit)
POSE_READY = [0.05, -0.05, -0.05, 0.1, 0, 0]

# 3. Posisi Grasp (Turun ke bawah / Nungging)
POSE_GRASP = [0.19, -0.928, -0.765, 0.2, 0, 0]

# 4. Posisi Angkat (Kembali ke atas membawa kubus)
POSE_LIFT = [0.05, -0.05, -0.05, 0.1, 0, 0]

# --- KONTROL GRIPPER ---
# Index actuator gripper biasanya ada di urutan terakhir (6 dan 7)
# Range di XML Anda: -0.7 s/d 0 (Kiri) dan 0 s/d 0.7 (Kanan)
GRIPPER_OPEN = [-0.7, 0.7]  # Buka lebar
GRIPPER_CLOSE = [-0.02, 0.02]  # Tutup (jepit kubus 5cm)


def move_smoothly(model, data, target_joints, target_gripper, duration, viewer):
    """
    Menggerakkan robot dari posisi sekarang ke target secara halus (interpolasi).
    """
    start_joints = data.ctrl[:6].copy()
    start_gripper = data.ctrl[6:8].copy()

    frames = int(duration * 60)  # 60 FPS

    for i in range(frames):
        alpha = i / frames  # 0.0 sampai 1.0

        # Rumus Interpolasi Linear: Current = Start + (Target - Start) * alpha
        new_joints = start_joints + (target_joints - start_joints) * alpha
        new_gripper = start_gripper + (target_gripper - start_gripper) * alpha

        # Kirim ke motor
        data.ctrl[:6] = new_joints
        data.ctrl[6:8] = new_gripper

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def main():
    print(f"Memuat model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Inisialisasi posisi awal (Home)
    data.qpos[:6] = POSE_HOME
    data.ctrl[:6] = POSE_HOME
    data.ctrl[6:8] = GRIPPER_OPEN
    mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Mulai Simulasi FK...")
        time.sleep(1)  # Jeda sebentar

        # 1. Gerak ke posisi READY
        print("-> Menuju Posisi Ready...")
        move_smoothly(
            model, data, POSE_READY, GRIPPER_OPEN, duration=2.0, viewer=viewer
        )

        # 2. Turun ke posisi GRASP (Mengambil)
        print("-> Turun Mengambil Kubus...")
        move_smoothly(
            model, data, POSE_GRASP, GRIPPER_OPEN, duration=2.0, viewer=viewer
        )

        # 3. Menutup Gripper (Tanpa gerak sendi lengan)
        print("-> Menjepit...")
        move_smoothly(
            model, data, POSE_GRASP, GRIPPER_CLOSE, duration=2.0, viewer=viewer
        )
        time.sleep(2.0)  # Tunggu fisik gripper mencengkeram

        # 4. Angkat ke posisi LIFT
        print("-> Mengangkat!")
        move_smoothly(
            model, data, POSE_LIFT, GRIPPER_CLOSE, duration=2.0, viewer=viewer
        )

        print("Selesai! Tekan Ctrl+C untuk keluar.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
