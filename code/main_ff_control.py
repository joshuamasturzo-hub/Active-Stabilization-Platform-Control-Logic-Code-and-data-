"""Feedforward + Proportional stabilization at 200 Hz.

Control law
-----------
    erpm = SIGN * (Kp * imu1_angle + Kff * imu2_rate)

  - P term (Kp * imu1_angle):  reacts to current tilt — same as before
  - FF term (Kff * imu2_rate): reacts to how fast the BOAT is moving,
    before the platform has even tilted. IMU2 is bolted to the boat so
    its angular rate IS the incoming disturbance. This eliminates the
    fundamental lag of pure feedback control.

Loop rate
---------
  200 Hz (5 ms period) vs previous 50 Hz (20 ms).
  Reduces control latency from 20 ms to 5 ms.
  Telemetry is read every cycle but motors are commanded every cycle.

Tuning guide
------------
  Kp    — same as before, start at proven values (Roll=600, Pitch=300)
  Kff   — start small (5-10). If FF overcorrects (platform tilts opposite
          to boat), reduce Kff. If still lagging, increase Kff.
  Kff sign — if FF makes things worse (amplifies tilt), negate it.

Hardware
--------
  IMU1: BNO08x on I2C bus 7  (platform — drives feedback)
  IMU2: BNO08x on I2C bus 1  (boat reference — drives feedforward)
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_ff_control.py
"""

import csv
import datetime
import sys
import time

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

# ── Port Configuration ────────────────────────────────────────────────────────

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"

# ── Control Parameters ────────────────────────────────────────────────────────

LOOP_PERIOD: float = 0.005   # 200 Hz

# P gains — proven stable baseline
ROLL_GAIN:  float = 600.0
PITCH_GAIN: float = 300.0

# Feedforward gains — IMU2 angular rate (deg/s) to ERPM.
# Start conservative. At 30 deg/s boat motion: FF contribution = 30 * Kff.
# Increase if platform still lags, decrease if FF overcorrects.
ROLL_KFF:  float = 70.0
PITCH_KFF: float = 40.0

# EMA filter on IMU2 rates before feedforward uses them.
# alpha=0.4 at 200Hz ~ 8Hz cutoff. Prevents FF from reacting to noise.
FF_FILTER_ALPHA: float = 0.4

# Motor direction signs
SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

MAX_ERPM: float = 5000.0

ROLL_DEADBAND_DEG:  float = 0.5
PITCH_DEADBAND_DEG: float = 1.0

MAX_MOTOR_AMPS:       float = 4.5
MAX_OVERCURRENT_CYCLES: int = 15   # ~75 ms at 200 Hz

# ── Hardware Initialisation ───────────────────────────────────────────────────

print("[INFO] Initialising IMU1 (bus 7)...")
imu1 = IMUReader(i2c_bus=7)
print("[INFO] IMU1 ready.")

try:
    imu2 = IMUReader(i2c_bus=1)
    imu2_available = True
    print("[INFO] IMU2 (bus 1) ready.")
except Exception as e:
    imu2 = None
    imu2_available = False
    print(f"[WARN] IMU2 not available — feedforward disabled: {e}")

motor_roll  = SerialMotorDriver(port=PORT_ROLL,  motor_id=1)
motor_pitch = SerialMotorDriver(port=PORT_PITCH, motor_id=2)

roll_ok  = motor_roll.connect()
pitch_ok = motor_pitch.connect()

if not roll_ok:
    print(f"[WARN] Roll motor not connected ({PORT_ROLL}).", file=sys.stderr)
if not pitch_ok:
    print(f"[WARN] Pitch motor not connected ({PORT_PITCH}).", file=sys.stderr)

if roll_ok:
    motor_roll.arm()
    print(f"[INFO] Roll  motor armed on {PORT_ROLL}")
if pitch_ok:
    motor_pitch.arm()
    print(f"[INFO] Pitch motor armed on {PORT_PITCH}")

print("[INFO] Waiting 3 s for IMU to stabilise...")
for i in range(3, 0, -1):
    a = imu1.get_angles()
    print(f"  {i}s  roll:{a['roll']:+6.2f} deg  pitch:{a['pitch']:+6.2f} deg", end="\r")
    time.sleep(1.0)
print("\n[INFO] Ready.")
print(f"[INFO] FF+P mode | ROLL Kp={ROLL_GAIN} Kff={ROLL_KFF} | PITCH Kp={PITCH_GAIN} Kff={PITCH_KFF}")
print(f"[INFO] Loop={int(1/LOOP_PERIOD)} Hz | MAX_ERPM={MAX_ERPM} | DEADBAND roll={ROLL_DEADBAND_DEG} pitch={PITCH_DEADBAND_DEG}")
if not imu2_available:
    print("[WARN] Running P-only (no IMU2 for feedforward).")

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_FILE = f"ff_control_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s",
    "imu1_roll_deg", "imu1_pitch_deg",
    "imu2_roll_deg", "imu2_pitch_deg",
    "imu2_roll_rate_dps", "imu2_pitch_rate_dps",
    "ff_roll_erpm", "ff_pitch_erpm",
    "cmd_roll_erpm", "cmd_pitch_erpm",
    "motor_roll_erpm", "motor_pitch_erpm",
    "motor_roll_amps", "motor_pitch_amps",
    "dt_ms",
])
print(f"[INFO] Logging to {LOG_FILE}")
print("[INFO] Entering 200 Hz control loop.  Ctrl+C to stop.")

# ── 200 Hz Control Loop ───────────────────────────────────────────────────────

try:
    _roll_offset = _pitch_offset = 0.0
    if roll_ok:
        motor_roll.request_telemetry(); time.sleep(0.1)
        s = motor_roll.get_state()
        if s: _roll_offset = s["pos"]
    if pitch_ok:
        motor_pitch.request_telemetry(); time.sleep(0.1)
        s = motor_pitch.get_state()
        if s: _pitch_offset = s["pos"]

    prev_time: float = time.perf_counter()
    loop_count: int  = 0
    motor_roll_erpm  = motor_pitch_erpm = 0.0
    motor_roll_pos   = motor_pitch_pos  = 0.0
    motor_roll_amps  = motor_pitch_amps = 0.0
    filt_imu2_roll_rate  = 0.0
    filt_imu2_pitch_rate = 0.0
    overcurrent_cycles: int = 0

    while True:
        loop_start = time.perf_counter()
        dt = loop_start - prev_time
        prev_time = loop_start

        # ── IMU1 — platform angles (feedback) ────────────────────────
        angles1 = imu1.get_angles()
        roll    = angles1["roll"]
        pitch   = angles1["pitch"]

        # ── IMU2 — boat angles + rates (feedforward) ─────────────────
        imu2_roll = imu2_pitch = 0.0
        imu2_roll_rate = imu2_pitch_rate = 0.0
        if imu2_available:
            angles2 = imu2.get_angles()
            imu2_roll  = angles2["roll"]
            imu2_pitch = angles2["pitch"]
            rates2 = imu2.get_rates()
            imu2_roll_rate  = rates2["roll_rate"]
            imu2_pitch_rate = rates2["pitch_rate"]
            # EMA filter on IMU2 rates
            filt_imu2_roll_rate  = FF_FILTER_ALPHA * imu2_roll_rate  + (1 - FF_FILTER_ALPHA) * filt_imu2_roll_rate
            filt_imu2_pitch_rate = FF_FILTER_ALPHA * imu2_pitch_rate + (1 - FF_FILTER_ALPHA) * filt_imu2_pitch_rate

        # ── IMU Sanity Check ──────────────────────────────────────────
        if abs(roll) > 60.0 or abs(pitch) > 60.0:
            motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
            print(f"\n[FAULT] IMU out of range (roll={roll:.1f} pitch={pitch:.1f}) — stopped.")
            break

        # ── Control Law: P feedback + FF from boat rate ───────────────
        eff_roll  = roll  if abs(roll)  > ROLL_DEADBAND_DEG  else 0.0
        eff_pitch = pitch if abs(pitch) > PITCH_DEADBAND_DEG else 0.0

        ff_roll  = ROLL_KFF  * filt_imu2_roll_rate
        ff_pitch = PITCH_KFF * filt_imu2_pitch_rate

        cmd_roll  = max(-MAX_ERPM, min(SIGN_ROLL  * (ROLL_GAIN  * eff_roll  + ff_roll),  MAX_ERPM))
        cmd_pitch = max(-MAX_ERPM, min(SIGN_PITCH * (PITCH_GAIN * eff_pitch + ff_pitch), MAX_ERPM))

        # ── Send Commands ─────────────────────────────────────────────
        if roll_ok:  motor_roll.set_rpm(cmd_roll)
        if pitch_ok: motor_pitch.set_rpm(cmd_pitch)

        # ── Telemetry (every cycle at 200 Hz) ────────────────────────
        if roll_ok:  motor_roll.request_telemetry()
        if pitch_ok: motor_pitch.request_telemetry()

        if roll_ok:
            s = motor_roll.get_state()
            if s:
                motor_roll_erpm = s["rpm"]
                motor_roll_pos  = s["pos"] - _roll_offset
                motor_roll_amps = abs(s["torque"])
        if pitch_ok:
            s = motor_pitch.get_state()
            if s:
                motor_pitch_erpm = s["rpm"]
                motor_pitch_pos  = s["pos"] - _pitch_offset
                motor_pitch_amps = abs(s["torque"])

        # ── Current Protection ────────────────────────────────────────
        if motor_roll_amps > MAX_MOTOR_AMPS or motor_pitch_amps > MAX_MOTOR_AMPS:
            overcurrent_cycles += 1
            if overcurrent_cycles >= MAX_OVERCURRENT_CYCLES:
                motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
                print(f"\n[FAULT] Overcurrent — roll:{motor_roll_amps:.2f}A pitch:{motor_pitch_amps:.2f}A — stopped.")
                break
        else:
            overcurrent_cycles = 0

        # ── Terminal (print every 20 loops to avoid slowing 200 Hz) ──
        if loop_count % 20 == 0:
            imu2_str = f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}" if imu2_available else "IMU2:N/A"
            print(
                f"t={loop_count*LOOP_PERIOD:07.2f}s | "
                f"Roll:{roll:+6.2f}=>{cmd_roll:+5.0f}ERPM {motor_roll_amps:.2f}A | "
                f"Pitch:{pitch:+6.2f}=>{cmd_pitch:+5.0f}ERPM {motor_pitch_amps:.2f}A | "
                f"{imu2_str} | dt:{dt*1000:.1f}ms",
                end="\r",
            )

        # ── Log ───────────────────────────────────────────────────────
        log_csv.writerow([
            f"{loop_count*LOOP_PERIOD:.4f}",
            f"{roll:.4f}",       f"{pitch:.4f}",
            f"{imu2_roll:.4f}",  f"{imu2_pitch:.4f}",
            f"{filt_imu2_roll_rate:.3f}", f"{filt_imu2_pitch_rate:.3f}",
            f"{ff_roll:.1f}",    f"{ff_pitch:.1f}",
            f"{cmd_roll:.1f}",   f"{cmd_pitch:.1f}",
            f"{motor_roll_erpm:.0f}", f"{motor_pitch_erpm:.0f}",
            f"{motor_roll_amps:.3f}", f"{motor_pitch_amps:.3f}",
            f"{dt*1000:.2f}",
        ])

        loop_count += 1

        # ── Rate Limit ────────────────────────────────────────────────
        used = time.perf_counter() - loop_start
        sleep_time = LOOP_PERIOD - used
        if sleep_time > 0.0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted. Shutting down...")

finally:
    print("[INFO] Zeroing motors...")
    if roll_ok:  motor_roll.set_rpm(0)
    if pitch_ok: motor_pitch.set_rpm(0)
    time.sleep(0.3)
    motor_roll.stop()
    motor_pitch.stop()
    log_fh.close()
    print(f"[INFO] Stopped. Data saved to {LOG_FILE}.")
