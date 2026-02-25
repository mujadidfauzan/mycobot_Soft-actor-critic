import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    # 1. Muat model dan data dari file XML utama
    model_path = "/home/fauzan/Mujoco/Skripsi/source/robot/pickplace.xml"
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    # 2. Definisikan urutan waypoint (Target posisi untuk 8 aktuator)
    # Urutan aktuator berdasarkan file XML:
    # [link2_to_link1, link3_to_link2, link4_to_link3, link5_to_link4,
    #  link6_to_link5, link6output_to_link6, gripper_l, gripper_r]

    # Catatan: Nilai joint (radian) ini adalah perkiraan kasar (hardcoded).
    # Untuk posisi yang 100% akurat ke koordinat objek [0.27, 0, 0.1],
    # Anda idealnya menggunakan Inverse Kinematics (IK).

    waypoints = [
        {
            "name": "1. Posisi Awal (Home)",
            "ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02]),
            "duration": 1.0,  # Waktu tunggu agar robot mencapai posisi (detik)
        },
        {
            "name": "2. Mendekati Objek (Reach)",
            # Menekuk joint 2, 3, dan 4 ke bawah, gripper tetap terbuka
            "ctrl": np.array([0.176, -1.58, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02]),
            "duration": 2.0,
        },
        {
            "name": "3. Menutup Gripper (Grasp)",
            # Posisi lengan tetap, nilai gripper diset minus/rapat
            "ctrl": np.array([0.176, -1.58, 0.0, 0.0, 0.0, 0.0, -0.02, 0.02]),
            "duration": 2.0,
        },
        {
            "name": "4. Mengangkat Objek (Lift)",
            # Mengangkat joint 2 ke atas, menahan objek
            "ctrl": np.array([0.176, -1.2, 0.0, 0.0, 0.0, 0.0, -0.02, 0.02]),
            "duration": 2.0,
        },
    ]

    # 3. Jalankan simulasi dengan viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Sinkronisasi awal
        viewer.sync()

        for wp in waypoints:
            print(f"Fase: {wp['name']}")

            # Set target kontrol (Posisi PD Control)
            data.ctrl[:] = wp["ctrl"]

            # Hitung berapa banyak step fisika yang dibutuhkan untuk durasi ini
            # timestep default di XML adalah 0.002 detik
            steps = int(wp["duration"] / model.opt.timestep)

            for _ in range(steps):
                # Jalankan 1 step simulasi fisika
                mujoco.mj_step(model, data)

                # Render ke viewer secara berkala (tidak perlu setiap step fisika)
                # Sinkronisasi visual setiap ~1/60 detik agar tidak berat
                viewer.sync()

                # Tambahkan sedikit jeda agar simulasi berjalan sesuai waktu nyata (real-time)
                time.sleep(model.opt.timestep)

                # Cek apakah window ditutup oleh user
                if not viewer.is_running():
                    break

            if not viewer.is_running():
                break

    print("Simulasi selesai.")


if __name__ == "__main__":
    main()
