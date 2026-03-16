import json
import socket


class MyCobotRemote:
    """UDP client for communicating with a remote MyCobot robot arm."""

    DEFAULT_PORT = 5005
    DEFAULT_TIMEOUT = 0.1
    NUM_JOINTS = 6

    def __init__(
        self, ip: str, port: int = DEFAULT_PORT, timeout: float = DEFAULT_TIMEOUT
    ):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

        # Last known good state (fallback when communication fails)
        self._last_angles: list[float] = [0.0] * self.NUM_JOINTS
        self._last_coords: list[float] = [0.0] * self.NUM_JOINTS

    def _send(self, command: str, data=None) -> dict | None:
        """Send a UDP command and return the parsed response (or None on failure)."""
        message = json.dumps({"command": command, "data": data}).encode()
        try:
            self.sock.sendto(message, self.addr)
            if command == "GET_STATE":
                raw, _ = self.sock.recvfrom(1024)
                return json.loads(raw.decode())
            return {}
        except Exception:
            return None

    def update_state(self) -> None:
        """
        Fetch angles + coords in ONE UDP round-trip and cache them.
        Call this ONCE per control loop, then read .angles and .coords directly
        to avoid multiple blocking network calls per iteration.
        """
        state = self._send("GET_STATE") or {}
        if state.get("angles"):
            self._last_angles = state["angles"]
        if state.get("coords"):
            self._last_coords = state["coords"]

    @property
    def angles(self) -> list[float]:
        """Last cached joint angles in degrees."""
        print(f"Current angles: {self._last_angles}")
        return self._last_angles

    @property
    def coords(self) -> list[float]:
        """Last cached end-effector coords [x, y, z, rx, ry, rz]."""
        return self._last_coords

    # Keep these for backward compatibility — but note each fires a UDP call
    def get_angles(self) -> list[float]:
        self.update_state()
        return self._last_angles

    def get_coords(self) -> list[float]:
        return self._last_coords  # reuse state already fetched by update_state()

    def send_angles(self, angles: list[float], speed: int = 20) -> bool:
        """Send target joint angles (degrees) at the given speed."""
        return self._send("SET_ANGLES", [float(a) for a in angles]) is not None

    def set_gripper_state(self, state: int, speed: int = 50) -> bool:
        """Set gripper state: 0 = open, 1 = closed."""
        return self._send("SET_GRIPPER", state) is not None

    def power_on(self) -> None:
        print(f"Connected to robot at {self.addr}")

    def stop(self) -> None:
        print("Stopping remote connection...")
        self.sock.close()
