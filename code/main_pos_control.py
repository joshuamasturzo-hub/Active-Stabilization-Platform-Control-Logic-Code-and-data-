"""Position-loop stabilization script for the marine platform.

Strategy
--------
Instead of commanding torque (which requires careful PID tuning and has
derivative-noise problems), we use the motor controller's built-in
position PID loop.  We simply tell each motor WHERE to be in degrees, and
the motor firmware drives as much current as necessary to reach and hold
that point.

Control law (per axis):
    target_pos_deg = clamp(GAIN * imu_angle, -POS_LIMIT_DEG, +POS_LIMIT_DEG)

Where:
  - imu_angle is the tilt in degrees (0 = level).
  - GAIN = 1.0 means 1 deg of tilt -> 1 deg of motor correction.
  - A higher GAIN gives faster/stiffer correction.
  - Sign conventions match the torque-control scripts (see SIGN_ROLL /
    SIGN_PITCH below). Flip these if the motor fights the tilt instead of
    correcting it.

Startup sequence
----------------
1. Connect motors, connect IMUs.
2. Wait 3 s for IMUs to stabilise.
3. Set motor origin (temporary) — THE critical step that was missing before
   and caused the previous spinout.
4. Enter 50 Hz control loop.

Hardware
--------
  IMU1:  BNO08x on I2C bus 7, address 0x4A  (primary — drives control)
  IMU2:  BNO08x on I2C bus 1, address 0x4A  (verification — log only)
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL

Usage:
    python3 main_pos_control.py
"""

import csv
import sys
import time

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

# ── Port Configuration ────────────────────────────────────────────────────────

PORT_PITCH: str = "/dev/ttyPITCH"
PORT_ROLL: str  = "/dev/ttyROLL"

# ── Control Parameters ────────────────────────────────────────────────────────

LOOP_PERIOD: float = 0.02       # 50 Hz

# Proportional gain: how many degrees the motor moves per degree of tilt.
# Start at 1.0 (direct tracking).  Increase for stiffer hold, decrease if
# the platform oscillates.
ROLL_GAIN:  float = 0.4
PITCH_GAIN: float = 0.4

# Sign of the correction.  If the motor fights the tilt (makes it worse),
# flip the relevant sign to -1.
# Determined from torque-mode sign conventions:
#   - Roll torque was NOT negated  => motor moves negative to correct positive roll
#   - Pitch torque WAS negated     => motor moves positive to correct positive pitch
SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

# Hard mechanical limit: motor will never be commanded beyond +-45 deg from origin.
POS_LIMIT_DEG: float = 30.0

# Small deadband — angles smaller than this are treated as zero to prevent
# motor hunting at equilibrium.
DEADBAND_DEG: float = 0.5

# Speed and acceleration for position-velocity loop (smoother than raw position).
# speed_erpm: electrical RPM limit (AK60-39 has 21 pole pairs; 12000 ERPM ~ 27 rpm shaft)
# accel: electrical acceleration limit
POS_SPEED_ERPM: int = 6000
POS_ACCEL:      int = 20000

# ── Hardware Initialisation ───────────────────────────────────────────────────

print("[INFO] Initialising IMU1 (bus 7)...")
imu1 = IMUReader(i2c_bus=7)
print("[INFO] IMU1 ready.")

try:
    imu2 = IMUReader(i2c_bus=1)
    imu2_available = True
    print("[INFO] IMU2 (bus 1) ready — verification logging active.")
except Exception as e:
    imu2 = None
    imu2_available = False
    print(f"[WARN] IMU2 (bus 1) not available: {e}")

motor_roll  = SerialMotorDriver(port=PORT_ROLL,  motor_id=1)
motor_pitch = SerialMotorDriver(port=PORT_PITCH, motor_id=2)

roll_ok  = motor_roll.connect()
pitch_ok = motor_pitch.connect()

if not roll_ok:
    print(f"[WARN] Roll motor not connected ({PORT_ROLL}) — roll axis disabled.",
          file=sys.stderr)
if not pitch_ok:
    print(f"[WARN] Pitch motor not connected ({PORT_PITCH}) — pitch axis disabled.",
          file=sys.stderr)

if roll_ok:
    motor_roll.arm()
    print(f"[INFO] Roll  motor armed on {PORT_ROLL}")
if pitch_ok:
    motor_pitch.arm()
    print(f"[INFO] Pitch motor armed on {PORT_PITCH}")

# ── IMU Stabilisation Wait ────────────────────────────────────────────────────

print("[INFO] Waiting 3 s for IMU sensors to stabilise...")
for i in range(3, 0, -1):
    angles_preview = imu1.get_angles()
    print(f"  {i}s — IMU1 roll:{angles_preview['roll']:+6.2f} deg  "
          f"pitch:{angles_preview['pitch']:+6.2f} deg", end="\r")
    time.sleep(1.0)
print("\n[INFO] IMU stabilised.")

# ── SET ORIGIN — Zero the motors at their current shaft position ──────────────
# Without this the motor remembers an arbitrary position from power-on and will
# immediately spin hard to reach it when the first position command is sent.

print("[INFO] Setting motor origins (temporary zero at current shaft position)...")
if roll_ok:
    motor_roll.set_origin(permanent=False)
if pitch_ok:
    motor_pitch.set_origin(permanent=False)
time.sleep(0.2)   # let both controllers acknowledge before sending position cmds
print("[INFO] Origins set.")
print(f"[INFO] POS_LIMIT=+-{POS_LIMIT_DEG} deg (hard limit)  DEADBAND={DEADBAND_DEG} deg  "
      f"ROLL_GAIN={ROLL_GAIN}  PITCH_GAIN={PITCH_GAIN}")
print(f"[INFO] Speed limit={POS_SPEED_ERPM} ERPM  Accel limit={POS_ACCEL}")

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_FILE = "pos_control_data.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s",
    "imu1_roll_deg", "target_roll_deg",
    "imu1_pitch_deg", "target_pitch_deg",
    "imu2_roll_deg", "imu2_pitch_deg",
    "motor_roll_pos_deg", "motor_pitch_pos_deg",
    "dt_ms",
])
print(f"[INFO] Logging to {LOG_FILE}")
print("[INFO] Entering 50 Hz control loop.  Press Ctrl+C to stop.")

# ── 50 Hz Control Loop ────────────────────────────────────────────────────────

try:
    prev_time: float = time.perf_counter()
    loop_count: int  = 0

    motor_roll_pos:  float = 0.0
    motor_pitch_pos: float = 0.0

    while True:
        loop_start: float = time.perf_counter()
        dt: float = loop_start - prev_time
        prev_time = loop_start

        # ── IMU Read ─────────────────────────────────────────────────
        angles1 = imu1.get_angles()
        roll  = angles1["roll"]
        pitch = angles1["pitch"]

        imu2_roll  = 0.0
        imu2_pitch = 0.0
        if imu2_available:
            angles2 = imu2.get_angles()
            imu2_roll  = angles2["roll"]
            imu2_pitch = angles2["pitch"]

        # ── Position Target Computation ───────────────────────────────
        # Apply deadband to avoid hunting at equilibrium.
        eff_roll  = roll  if abs(roll)  > DEADBAND_DEG else 0.0
        eff_pitch = pitch if abs(pitch) > DEADBAND_DEG else 0.0

        # Motor must move opposite (or proportional) to the tilt.
        raw_roll_target  = SIGN_ROLL  * ROLL_GAIN  * eff_roll
        raw_pitch_target = SIGN_PITCH * PITCH_GAIN * eff_pitch

        # Hard clamp — never exceed +-POS_LIMIT_DEG from origin.
        target_roll  = max(-POS_LIMIT_DEG, min(raw_roll_target,  POS_LIMIT_DEG))
        target_pitch = max(-POS_LIMIT_DEG, min(raw_pitch_target, POS_LIMIT_DEG))

        # ── Motor Position Commands ───────────────────────────────────
        if roll_ok:
            motor_roll.set_pos_spd(target_roll,  speed_erpm=POS_SPEED_ERPM, accel=POS_ACCEL)
        if pitch_ok:
            motor_pitch.set_pos_spd(target_pitch, speed_erpm=POS_SPEED_ERPM, accel=POS_ACCEL)

        # ── Request Telemetry for Next Cycle ──────────────────────────
        if roll_ok:
            motor_roll.request_telemetry()
        if pitch_ok:
            motor_pitch.request_telemetry()

        # ── Motor Feedback ────────────────────────────────────────────
        if roll_ok:
            state_roll = motor_roll.get_state()
            if state_roll is not None:
                motor_roll_pos = state_roll["pos"]

        if pitch_ok:
            state_pitch = motor_pitch.get_state()
            if state_pitch is not None:
                motor_pitch_pos = state_pitch["pos"]

        # ── Terminal Status ───────────────────────────────────────────
        imu2_str = (
            f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}"
            if imu2_available else "IMU2:N/A"
        )
        print(
            f"t={loop_count * LOOP_PERIOD:07.2f}s | "
            f"Roll:{roll:+6.2f}->{target_roll:+6.2f} | "
            f"Pitch:{pitch:+6.2f}->{target_pitch:+6.2f} | "
            f"MtrR:{motor_roll_pos:+6.1f} MtrP:{motor_pitch_pos:+6.1f} | "
            f"{imu2_str} | dt:{dt * 1000:.1f}ms",
            end="\r",
        )

        # ── CSV Log ───────────────────────────────────────────────────
        log_csv.writerow([
            f"{loop_count * LOOP_PERIOD:.3f}",
            f"{roll:.4f}",        f"{target_roll:.4f}",
            f"{pitch:.4f}",       f"{target_pitch:.4f}",
            f"{imu2_roll:.4f}",   f"{imu2_pitch:.4f}",
            f"{motor_roll_pos:.4f}",
            f"{motor_pitch_pos:.4f}",
            f"{dt * 1000:.2f}",
        ])

        loop_count += 1

        # ── Rate Limiting ─────────────────────────────────────────────
        used: float = time.perf_counter() - loop_start
        sleep_time: float = LOOP_PERIOD - used
        if sleep_time > 0.0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by operator. Shutting down...")

finally:
    # Slowly return motors to origin before cutting power.
    print("[INFO] Returning motors to origin (0 deg)...")
    if roll_ok:
        motor_roll.set_pos_spd(0.0, speed_erpm=5000, accel=20000)
    if pitch_ok:
        motor_pitch.set_pos_spd(0.0, speed_erpm=5000, accel=20000)
    time.sleep(0.5)
    motor_roll.stop()
    motor_pitch.stop()
    log_fh.close()
    print(f"[INFO] Motors stopped. Data saved to {LOG_FILE}.")
