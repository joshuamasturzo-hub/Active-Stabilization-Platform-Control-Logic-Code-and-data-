"""Main control loop — dual IMU edition.

Identical to main.py but reads both IMUs and logs them side-by-side for
cross-validation. IMU1 (bus 7) drives the control loop; IMU2 (bus 1) is
logged only and never fed into the controller.

Hardware:
    - IMU1:  BNO08x on I2C bus 7, address 0x4A  (primary — control)
    - IMU2:  BNO08x on I2C bus 1, address 0x4A  (verification — log only)
    - Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH  (udev: physical port 1-2.1)
    - Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL   (udev: physical port 1-2.2)

CSV columns:
    t_s, roll_deg, cmd_roll_A, pitch_deg, cmd_pitch_A,
    imu2_roll_deg, imu2_pitch_deg, dt_ms

Usage:
    python3 main_dual_imu.py
"""

import csv
import sys
import time

from imu_sensor import IMUReader
from pd_controller import PDController
from serial_motor_driver import SerialMotorDriver

# ── Port Configuration ───────────────────────────────────────────────────────

PORT_PITCH: str = "/dev/ttyPITCH"  # physical port 1-2.1, udev symlink
PORT_ROLL: str  = "/dev/ttyROLL"   # physical port 1-2.2, udev symlink

# ── Control Parameters ───────────────────────────────────────────────────────

LOOP_PERIOD: float = 0.02     # 50 Hz
TARGET_ANGLE: float = 0.0     # Level-hold setpoint (degrees)

# ── Master State Dictionary ──────────────────────────────────────────────────

State: dict = {
    "IMU": {
        "roll": 0.0,
        "pitch": 0.0,
    },
    "IMU2": {
        "roll": 0.0,
        "pitch": 0.0,
    },
    "Motors": {
        1: {"torque": 0.0, "pos": 0.0},
        2: {"torque": 0.0, "pos": 0.0},
    },
    "Tuning": {
        "roll_Kp":  0.25,
        "roll_Ki":  0.05,
        "roll_Kd":  0.008,
        "pitch_Kp": 0.12,
        "pitch_Ki": 0.03,
        "pitch_Kd": 0.015,
        "max_torque": 3.0,
    },
}

# ── Hardware Initialisation ──────────────────────────────────────────────────

imu1 = IMUReader(i2c_bus=7)

try:
    imu2 = IMUReader(i2c_bus=7, address=0x4A)
    imu2_available = True
    print("[INFO] IMU2 (bus 1) initialised — verification logging active.")
except Exception as e:
    imu2 = None
    imu2_available = False
    print(f"[WARN] IMU2 (bus 1) not available: {e} — imu2 columns will be 0.0.")

motor_roll = SerialMotorDriver(port=PORT_ROLL, motor_id=1)
motor_pitch = SerialMotorDriver(port=PORT_PITCH, motor_id=2)

if not motor_roll.connect():
    print(f"[FATAL] Failed to open UART for motor 1 (roll) on {PORT_ROLL}.",
          file=sys.stderr)

if not motor_pitch.connect():
    print(f"[FATAL] Failed to open UART for motor 2 (pitch) on {PORT_PITCH}.",
          file=sys.stderr)
    motor_roll.stop()

if motor_roll.ser is not None:
    print(f"[INFO] Roll  motor connected on {PORT_ROLL}")
else:
    print(f"[WARN] Roll  motor NOT connected ({PORT_ROLL}) — roll axis disabled.")
if motor_pitch.ser is not None:
    print(f"[INFO] Pitch motor connected on {PORT_PITCH}")
else:
    print(f"[WARN] Pitch motor NOT connected ({PORT_PITCH}) — pitch axis disabled.")

if motor_roll.ser is not None:
    motor_roll.arm()
if motor_pitch.ser is not None:
    motor_pitch.arm()
print("[INFO] Motors armed.")

# Allow IMUs to stabilize before control starts.
print("[INFO] Waiting 3s for IMU sensors to stabilize...")
for i in range(3, 0, -1):
    print(f"  {i}...", end="\r")
    time.sleep(1.0)
print("[INFO] IMU stabilized.                ")

# Record starting positions — used for the position trip-wire.
# If a motor travels more than POSITION_LIMIT degrees from its start,
# torque is cut for that axis to prevent a full rotation.
POSITION_LIMIT: float = 360.0
motor_roll.request_telemetry();  time.sleep(0.1)
motor_pitch.request_telemetry(); time.sleep(0.1)
_rs = motor_roll.get_state();  roll_pos_start  = _rs["pos"] if _rs else 0.0
_ps = motor_pitch.get_state(); pitch_pos_start = _ps["pos"] if _ps else 0.0
print(f"[INFO] Start positions — Roll: {roll_pos_start:.1f}°  Pitch: {pitch_pos_start:.1f}°")

# ── Controller Instantiation ─────────────────────────────────────────────────

ctrl_roll = PDController(
    Kp=State["Tuning"]["roll_Kp"],
    Ki=State["Tuning"]["roll_Ki"],
    Kd=State["Tuning"]["roll_Kd"],
    max_torque=State["Tuning"]["max_torque"],
)

ctrl_pitch = PDController(
    Kp=State["Tuning"]["pitch_Kp"],
    Ki=State["Tuning"]["pitch_Ki"],
    Kd=State["Tuning"]["pitch_Kd"],
    max_torque=State["Tuning"]["max_torque"],
)

# ── 50 Hz Control Loop ──────────────────────────────────────────────────────

LOG_FILE = "dual_imu_data.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s", "roll_deg", "cmd_roll_A", "pitch_deg", "cmd_pitch_A",
    "imu2_roll_deg", "imu2_pitch_deg", "dt_ms",
])
print(f"[INFO] Logging to {LOG_FILE}")
print(f"[INFO] Entering control loop at {1.0 / LOOP_PERIOD:.0f} Hz. Press Ctrl+C to stop.")

try:
    prev_time: float = time.perf_counter()
    loop_count: int = 0

    while True:
        loop_start: float = time.perf_counter()
        dt: float = loop_start - prev_time
        prev_time = loop_start

        # ── IMU Read ─────────────────────────────────────────────────
        angles = imu1.get_angles()
        State["IMU"]["roll"]  = angles["roll"]
        State["IMU"]["pitch"] = angles["pitch"]

        if imu2_available:
            angles2 = imu2.get_angles()
            State["IMU2"]["roll"]  = angles2["roll"]
            State["IMU2"]["pitch"] = angles2["pitch"]

        # ── PD Control (IMU1 only) ────────────────────────────────────
        # IMU mounted face-down: negate both angles to correct sign inversion.
        cmd_roll: float  = ctrl_roll.calculate(TARGET_ANGLE, angles["roll"], dt)
        cmd_pitch: float = ctrl_pitch.calculate(TARGET_ANGLE, angles["pitch"], dt)

        # ── Position Trip-wire ────────────────────────────────────────
        # Cut torque if motor has travelled more than POSITION_LIMIT from start.
        roll_travel  = abs(State["Motors"][1]["pos"] - roll_pos_start)
        pitch_travel = abs(State["Motors"][2]["pos"] - pitch_pos_start)
        if roll_travel > POSITION_LIMIT:
            print(f"\n[SAFETY] Roll motor exceeded {POSITION_LIMIT}° travel — axis disabled.")
            cmd_roll = 0.0
        if pitch_travel > POSITION_LIMIT:
            print(f"\n[SAFETY] Pitch motor exceeded {POSITION_LIMIT}° travel — axis disabled.")
            cmd_pitch = 0.0

        # ── Motor Output ─────────────────────────────────────────────
        motor_roll.send_torque(cmd_roll)
        motor_pitch.send_torque(cmd_pitch)

        # ── Request Telemetry for Next Cycle ─────────────────────────
        motor_roll.request_telemetry()
        motor_pitch.request_telemetry()

        # ── Motor Feedback ───────────────────────────────────────────
        state_roll = motor_roll.get_state()
        if state_roll is not None:
            State["Motors"][1]["torque"] = state_roll["torque"]
            State["Motors"][1]["pos"]    = state_roll["pos"]

        state_pitch = motor_pitch.get_state()
        if state_pitch is not None:
            State["Motors"][2]["torque"] = state_pitch["torque"]
            State["Motors"][2]["pos"]    = state_pitch["pos"]

        # ── Terminal Status ──────────────────────────────────────────
        imu2_str = (
            f"IMU2 R:{State['IMU2']['roll']:+6.2f}° P:{State['IMU2']['pitch']:+6.2f}°"
            if imu2_available else "IMU2:N/A"
        )
        print(
            f"t={loop_count * LOOP_PERIOD:07.2f}s | "
            f"Roll:{State['IMU']['roll']:+7.2f}° cmd:{cmd_roll:+6.3f}A | "
            f"Pitch:{State['IMU']['pitch']:+7.2f}° cmd:{cmd_pitch:+6.3f}A | "
            f"{imu2_str} | dt:{dt * 1000:5.1f}ms",
            end="\r",
        )

        # ── CSV Log ──────────────────────────────────────────────────
        log_csv.writerow([
            f"{loop_count * LOOP_PERIOD:.3f}",
            f"{angles['roll']:.4f}",
            f"{cmd_roll:.4f}",
            f"{angles['pitch']:.4f}",
            f"{cmd_pitch:.4f}",
            f"{State['IMU2']['roll']:.4f}",
            f"{State['IMU2']['pitch']:.4f}",
            f"{dt * 1000:.2f}",
        ])

        loop_count += 1

        # ── Rate Limiting ────────────────────────────────────────────
        used: float = time.perf_counter() - loop_start
        sleep_time: float = LOOP_PERIOD - used
        if sleep_time > 0.0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by operator. Shutting down...")

finally:
    motor_roll.stop()
    motor_pitch.stop()
    log_fh.close()
    print(f"[INFO] Motors stopped. Data saved to {LOG_FILE}.")
