"""Feedforward + PD stabilization at 200 Hz.

Control law
-----------
    erpm = SIGN * (Kp * imu1_angle + Kd * imu1_rate + Kff * imu2_rate)

  - P term  (Kp * imu1_angle):  reacts to current tilt
  - D term  (Kd * imu1_rate):   damps platform's own motion — suppresses
                                  the 2.8 Hz self-oscillation seen in FF+P tests.
                                  Reads from IMU1 gyroscope (platform rate).
  - FF term (Kff * imu2_rate):  anticipates boat disturbance before platform
                                  tilts. Reads from IMU2 gyroscope (boat rate).

Why D was added
---------------
FF+P tests showed 66-73% roll reduction during active waves but P self-oscillated
at 2.8 Hz whenever waves stopped (Roll gain = 600 is at the P-only stability
ceiling). D damps that oscillation and may allow P gains to be raised later.

Tuning guide
------------
  Start with these conservative D gains. If self-oscillation is gone, try
  raising Kp (Roll 600->700). If D makes things worse (oscillates faster),
  reduce D_FILTER_ALPHA or halve Kd.

  D_FILTER_ALPHA=0.15 is tighter than the 0.3 used in main_pd_control.py.
  The old value picked up motor vibration noise. 0.15 attenuates that harder.

  MAX_D_ERPM caps D so a noise spike can never run away.

Hardware
--------
  IMU1: BNO08x on I2C bus 7  (platform — P + D feedback)
  IMU2: BNO08x on I2C bus 1  (boat reference — feedforward)
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_fpd_control.py
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
PITCH_GAIN: float = 250.0

# D gains — IMU1 gyro rate (deg/s) to ERPM. Conservative start.
# At 10 deg/s platform rate: D = 10 * Kd = 100 ERPM roll, 50 ERPM pitch.
# If self-oscillation is still present, double these. If D causes faster
# oscillation, halve them or reduce D_FILTER_ALPHA further.
ROLL_KD:  float = 10.0
PITCH_KD: float = -5.0   # negative: SIGN_PITCH=+1 so KD must be negative to damp

# Hard cap on D contribution — prevents noise spikes from running away.
MAX_D_ERPM: float = 600.0

# EMA filter on IMU1 rates before D term uses them.
# 0.15 = tighter than main_pd_control.py (0.30) to reject motor vibration.
D_FILTER_ALPHA: float = 0.15

# Feedforward gains — IMU2 angular rate (deg/s) to ERPM. Proven working.
ROLL_KFF:  float = 70.0
PITCH_KFF: float = 40.0

# EMA filter on IMU2 rates before FF term uses them.
FF_FILTER_ALPHA: float = 0.4

# Motor direction signs
SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

MAX_ERPM: float = 5000.0

ROLL_DEADBAND_DEG:  float = 0.5
PITCH_DEADBAND_DEG: float = 1.0

MAX_MOTOR_AMPS:       float = 10.0
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
print(f"[INFO] FF+PD | ROLL  Kp={ROLL_GAIN} Kd={ROLL_KD}  Kff={ROLL_KFF}")
print(f"[INFO]       | PITCH Kp={PITCH_GAIN} Kd={PITCH_KD} Kff={PITCH_KFF}")
print(f"[INFO] MAX_ERPM={MAX_ERPM} | MAX_D_ERPM={MAX_D_ERPM} | D_alpha={D_FILTER_ALPHA} | FF_alpha={FF_FILTER_ALPHA}")
if not imu2_available:
    print("[WARN] Running PD-only (no IMU2 for feedforward).")

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_FILE = f"fpd_control_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s",
    "imu1_roll_deg", "imu1_pitch_deg",
    "imu2_roll_deg", "imu2_pitch_deg",
    "imu1_roll_rate_dps", "imu1_pitch_rate_dps",
    "imu2_roll_rate_dps", "imu2_pitch_rate_dps",
    "d_roll_erpm", "d_pitch_erpm",
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
    filt_imu1_roll_rate  = 0.0   # D filter state (IMU1)
    filt_imu1_pitch_rate = 0.0
    filt_imu2_roll_rate  = 0.0   # FF filter state (IMU2)
    filt_imu2_pitch_rate = 0.0
    overcurrent_cycles: int = 0

    while True:
        loop_start = time.perf_counter()
        dt = loop_start - prev_time
        prev_time = loop_start

        # ── IMU1 — platform angles + rates (P + D feedback) ──────────
        angles1 = imu1.get_angles()
        roll    = angles1["roll"]
        pitch   = angles1["pitch"]
        rates1  = imu1.get_rates()
        imu1_roll_rate  = rates1["roll_rate"]
        imu1_pitch_rate = rates1["pitch_rate"]

        # EMA filter on IMU1 rates (D term)
        filt_imu1_roll_rate  = D_FILTER_ALPHA * imu1_roll_rate  + (1 - D_FILTER_ALPHA) * filt_imu1_roll_rate
        filt_imu1_pitch_rate = D_FILTER_ALPHA * imu1_pitch_rate + (1 - D_FILTER_ALPHA) * filt_imu1_pitch_rate

        # ── IMU2 — boat rates (feedforward) ──────────────────────────
        imu2_roll = imu2_pitch = 0.0
        imu2_roll_rate = imu2_pitch_rate = 0.0
        if imu2_available:
            angles2 = imu2.get_angles()
            imu2_roll  = angles2["roll"]
            imu2_pitch = angles2["pitch"]
            rates2 = imu2.get_rates()
            imu2_roll_rate  = rates2["roll_rate"]
            imu2_pitch_rate = rates2["pitch_rate"]
            # EMA filter on IMU2 rates (FF term)
            filt_imu2_roll_rate  = FF_FILTER_ALPHA * imu2_roll_rate  + (1 - FF_FILTER_ALPHA) * filt_imu2_roll_rate
            filt_imu2_pitch_rate = FF_FILTER_ALPHA * imu2_pitch_rate + (1 - FF_FILTER_ALPHA) * filt_imu2_pitch_rate

        # ── IMU Sanity Check ──────────────────────────────────────────
        if abs(roll) > 60.0 or abs(pitch) > 60.0:
            motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
            print(f"\n[FAULT] IMU out of range (roll={roll:.1f} pitch={pitch:.1f}) — stopped.")
            break

        # ── Control Law: P + D + FF ───────────────────────────────────
        eff_roll  = roll  if abs(roll)  > ROLL_DEADBAND_DEG  else 0.0
        eff_pitch = pitch if abs(pitch) > PITCH_DEADBAND_DEG else 0.0

        d_roll  = max(-MAX_D_ERPM, min(ROLL_KD  * filt_imu1_roll_rate,  MAX_D_ERPM))
        d_pitch = max(-MAX_D_ERPM, min(PITCH_KD * filt_imu1_pitch_rate, MAX_D_ERPM))

        ff_roll  = ROLL_KFF  * filt_imu2_roll_rate
        ff_pitch = PITCH_KFF * filt_imu2_pitch_rate

        cmd_roll  = max(-MAX_ERPM, min(SIGN_ROLL  * (ROLL_GAIN  * eff_roll  + d_roll  + ff_roll),  MAX_ERPM))
        cmd_pitch = max(-MAX_ERPM, min(SIGN_PITCH * (PITCH_GAIN * eff_pitch + d_pitch + ff_pitch), MAX_ERPM))

        # ── Send Commands ─────────────────────────────────────────────
        if roll_ok:  motor_roll.set_rpm(cmd_roll)
        if pitch_ok: motor_pitch.set_rpm(cmd_pitch)

        # ── Telemetry ─────────────────────────────────────────────────
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

        # ── Terminal (every 20 loops) ─────────────────────────────────
        if loop_count % 20 == 0:
            imu2_str = f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}" if imu2_available else "IMU2:N/A"
            print(
                f"t={loop_count*LOOP_PERIOD:07.2f}s | "
                f"Roll:{roll:+6.2f}=>{cmd_roll:+5.0f}ERPM(D:{d_roll:+4.0f}) {motor_roll_amps:.2f}A | "
                f"Pitch:{pitch:+6.2f}=>{cmd_pitch:+5.0f}ERPM(D:{d_pitch:+4.0f}) {motor_pitch_amps:.2f}A | "
                f"{imu2_str} | dt:{dt*1000:.1f}ms",
                end="\r",
            )

        # ── Log ───────────────────────────────────────────────────────
        log_csv.writerow([
            f"{loop_count*LOOP_PERIOD:.4f}",
            f"{roll:.4f}",       f"{pitch:.4f}",
            f"{imu2_roll:.4f}",  f"{imu2_pitch:.4f}",
            f"{filt_imu1_roll_rate:.3f}", f"{filt_imu1_pitch_rate:.3f}",
            f"{filt_imu2_roll_rate:.3f}", f"{filt_imu2_pitch_rate:.3f}",
            f"{d_roll:.1f}",     f"{d_pitch:.1f}",
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
