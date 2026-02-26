import time
from xml.parsers.expat import model

import mujoco
import mujoco.viewer
import numpy as np


def main():
    model_path = "/home/fauzan/Mujoco/Skripsi/source/robot/object_lift.xml"
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    joint_target = np.array(
        [
            0.43323181122751636,
            -1.0090628536208912,
            -0.9193142699937576,
            0.20175908294582942,
            0.22779390073164746,
            0.697362652491631,
        ]
    )

    joint_lift = np.array(
        [
            0.19913068567790296,
            -0.6305886738330566,
            -1.7506900944715555,
            1.3378228296165577,
            0.6195168721310196,
            2.6778849465640695,
        ]
    )

    waypoints = [
        {
            "name": "1. Posisi Awal (Home)",
            "ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02]),
            "duration": 0.2,
        },
        {
            "name": "2. Mendekati Objek (Reach)",
            "ctrl": np.array(
                [
                    joint_target[0],
                    joint_target[1],
                    joint_target[2],
                    joint_target[3],
                    joint_target[4],
                    joint_target[5],
                    0.02,
                    -0.02,
                ]
            ),
            "duration": 1.0,
        },
        {
            "name": "3. Menutup Gripper (Grasp)",
            "ctrl": np.array(
                [
                    joint_target[0],
                    joint_target[1],
                    joint_target[2],
                    joint_target[3],
                    joint_target[4],
                    joint_target[5],
                    -0.02,
                    0.02,
                ]
            ),
            "duration": 1.0,
        },
        {
            "name": "4. Mengangkat Objek (Lift)",
            "ctrl": np.array(
                [
                    joint_lift[0],
                    joint_lift[1],
                    joint_lift[2],
                    joint_lift[3],
                    joint_lift[4],
                    joint_lift[5],
                    -0.02,
                    0.02,
                ]
            ),
            "duration": 2.0,
        },
    ]

    # Ganti posisi objek
    data.joint("cube_joint").qpos = [0.27, 0.0, 0.1, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        for wp in waypoints:
            print(f"Fase: {wp['name']}")

            data.ctrl[:] = wp["ctrl"]

            # Hitung berapa banyak step fisika yang dibutuhkan untuk durasi ini
            # timestep default di XML adalah 0.002 detik
            steps = int(wp["duration"] / model.opt.timestep)

            for _ in range(steps):
                # Jalankan 1 step simulasi fisika
                mujoco.mj_step(model, data)

                # Sinkronisasi visual setiap ~1/60 detik
                viewer.sync()

                time.sleep(model.opt.timestep)

                if not viewer.is_running():
                    break

            if not viewer.is_running():
                break

    print("Simulasi selesai.")


if __name__ == "__main__":
    main()
