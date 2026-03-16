import json
import socket

from pymycobot import MyCobot280

# Inisialisasi Robot
mc = MyCobot280("/dev/ttyAMA0", 115200)

# Setup Socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

print("Bridge MyCobot Aktif [Port 5005]...")

while True:
    data, addr = sock.recvfrom(1024)
    payload = json.loads(data.decode())

    # CEK REQUEST DARI LAPTOP
    cmd = payload.get("command")

    if cmd == "GET_STATE":
        # Ambil data dari robot
        res = {"angles": mc.get_angles(), "coords": mc.get_coords()}
        sock.sendto(json.dumps(res).encode(), addr)

    elif cmd == "SET_ANGLES":
        # Terima sudut dan gerakkan
        cmd_data = payload.get("data")  # This is the nested dict with angles/speed/seq
        angles = cmd_data.get("angles")  # Extract just the angles list
        speed = cmd_data.get("speed", 50)
        seq = cmd_data.get("seq")  # Track sequence number to drop stale commands

        print(f"SET_ANGLES seq={seq}: {angles} @ speed={speed}")
        mc.send_angles(angles, speed)

    elif cmd == "SET_GRIPPER":
        # Kontrol Gripper (1=tutup, 0=buka)
        state = payload.get("data")
        mc.set_gripper_state(state, 50)
        mc.set_gripper_state(state, 50)
