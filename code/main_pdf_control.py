"""PD + Feedforward velocity-loop stabilization — experimental.

Uses IMU2 (boat/frame) as a feedforward signal to pre-correct BEFORE
the platform tilts.  This is the key to a near-flat platform.

IMU Usage
---------
  IMU1 (bus 7): mounted on the stabilized platform — feedback error signal.
  IMU2 (bus 1): mounted on the boat frame — feedforward signal.

At startup, IMU2 is sampled for 2 seconds to establish a DC offset so
that feedforward only reacts to CHANGES from the starting position.

Control law
-----------
    erpm_cmd = SIGN * (KP * platform_tilt + KD * platform_rate + KFF * boat_tilt_change)

Known issues / tuning notes
----------------------------
  1. KFF sensitivity: if the platform over-corrects, reduce KFF.
     If it under-corrects against wave disturbances, increase KFF.
  2. Gyro axis signs: if D term makes oscillation worse, negate KD.
  3. IMU2 DC offset: zeroed at startup over 2s. If boat static tilt
     changes significantly after startup, re-run the script.

Hardware
--------
  IMU1:  BNO08x on I2C bus 7, address 0x4A  (platform — feedback)
  IMU2:  BNO08x on I2C bus 1, address 0x4A  (boat frame — feedforward)
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_pdf_control.py
"""

import csv
import datetime
import sys
import time
import numpy as np

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"

LOOP_PERIOD: float = 0.02   # 50 Hz

# P gains — raised from 600/300 now that FF provides stability headroom.
ROLL_GAIN:  float = 800.0
PITCH_GAIN: float = 400.0

# D gains — gyro sign verified correct (corr=0.66/0.68 vs d(angle)/dt).
ROLL_KD:  float = 8.0
PITCH_KD: float = 8.0

# Hard cap on D contribution.
MAX_D_ERPM: float = 800.0

# EMA low-pass filter on gyro rate before D term.
RATE_FILTER_ALPHA: float = 0.3

# EMA low-pass filter on IMU1 angle — removes high-frequency vibration from
# the platform base settling after motion.  Cutoff ~5 Hz at 50 Hz loop rate.
# alpha=0.46 → passes waves (<2 Hz) but attenuates vibration (>5 Hz).
ANGLE_FILTER_ALPHA: float = 0.46

# Feedforward gains — uses IMU2 (boat/frame motion) to pre-correct BEFORE
# the platform tilts.  This is the key to a near-flat platform.
# KFF = GAIN means: treat boat tilt the same as platform tilt for correction.
# Start at 0.5x gain and tune up.  If platform over-corrects, reduce KFF.
ROLL_KFF:  float = 150.0   # 0.25 × ROLL_GAIN
PITCH_KFF: float = 75.0    # 0.25 × PITCH_GAIN

# Continuous IMU2 baseline update — tracks slow DC drift without reacting to waves.
# alpha=0.001 at 50Hz → time constant ~20s (much slower than wave motion at 0.5-1Hz).
# Replaces the fixed 2s snapshot so FF stays zero-centered as the boat slowly shifts.
IMU2_BASELINE_ALPHA: float = 0.005

SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

MAX_ERPM: float = 5000.0


ROLL_DEADBAND_DEG:  float = 0.5
PITCH_DEADBAND_DEG: float = 1.0

MAX_MOTOR_AMPS:       float = 4.0
MAX_OVERCURRENT_CYCLES: int = 3

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
    print(f"[WARN] IMU2 not available: {e}")

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
print(f"[INFO] PDF mode — ROLL P={ROLL_GAIN} D={ROLL_KD} FF={ROLL_KFF}  PITCH P={PITCH_GAIN} D={PITCH_KD} FF={PITCH_KFF}  MAX_ERPM={MAX_ERPM}")

LOG_FILE = f"pdf_control_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s",
    "imu1_roll_deg", "cmd_roll_erpm",
    "imu1_pitch_deg", "cmd_pitch_erpm",
    "imu2_roll_deg", "imu2_pitch_deg",
    "motor_roll_erpm", "motor_pitch_erpm",
    "motor_roll_pos_deg", "motor_pitch_pos_deg",
    "motor_roll_amps", "motor_pitch_amps",
    "roll_rate_dps", "pitch_rate_dps",
    "d_roll_erpm", "d_pitch_erpm",
    "ff_roll_erpm", "ff_pitch_erpm",
    "dt_ms",
])
print(f"[INFO] Logging to {LOG_FILE}")
print("[INFO] Entering 50 Hz PD+FF control loop.  Ctrl+C to stop.")

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

    # Sample IMU2 baseline for 2 seconds to get a good starting point,
    # then update continuously with a slow EMA (tau~20s) to track DC drift.
    imu2_roll_baseline = imu2_pitch_baseline = 0.0
    if imu2_available:
        print("[INFO] Sampling IMU2 baseline (2s)...")
        samples = []
        for _ in range(100):   # 100 × 20ms = 2s
            a = imu2.get_angles()
            samples.append((a["roll"], a["pitch"]))
            time.sleep(0.02)
        imu2_roll_baseline  = float(np.mean([s[0] for s in samples]))
        imu2_pitch_baseline = float(np.mean([s[1] for s in samples]))
        print(f"[INFO] IMU2 initial baseline — roll:{imu2_roll_baseline:+.2f}°  pitch:{imu2_pitch_baseline:+.2f}°")

    prev_time: float = time.perf_counter()
    loop_count: int  = 0
    motor_roll_erpm = motor_pitch_erpm = 0.0
    motor_roll_pos  = motor_pitch_pos  = 0.0
    motor_roll_amps = motor_pitch_amps = 0.0
    overcurrent_cycles: int = 0
    filt_roll_rate:  float = 0.0   # EMA-filtered gyro rates
    filt_pitch_rate: float = 0.0
    filt_roll:  float = 0.0        # EMA-filtered IMU1 angles (vibration filter)
    filt_pitch: float = 0.0

    while True:
        loop_start = time.perf_counter()
        dt = loop_start - prev_time
        prev_time = loop_start

        angles1 = imu1.get_angles()
        roll    = angles1["roll"]
        pitch   = angles1["pitch"]
        rates1     = imu1.get_rates()
        roll_rate  = rates1["roll_rate"]
        pitch_rate = rates1["pitch_rate"]

        # Vibration filter — smooths IMU1 angles before control law
        filt_roll  = ANGLE_FILTER_ALPHA * roll  + (1 - ANGLE_FILTER_ALPHA) * filt_roll
        filt_pitch = ANGLE_FILTER_ALPHA * pitch + (1 - ANGLE_FILTER_ALPHA) * filt_pitch
        roll  = filt_roll
        pitch = filt_pitch

        # EMA low-pass filter — smooths gyro noise before D term uses it
        filt_roll_rate  = RATE_FILTER_ALPHA * roll_rate  + (1 - RATE_FILTER_ALPHA) * filt_roll_rate
        filt_pitch_rate = RATE_FILTER_ALPHA * pitch_rate + (1 - RATE_FILTER_ALPHA) * filt_pitch_rate

        imu2_roll = imu2_pitch = 0.0
        if imu2_available:
            angles2    = imu2.get_angles()
            imu2_roll  = angles2["roll"]
            imu2_pitch = angles2["pitch"]

        if abs(roll) > 60.0 or abs(pitch) > 60.0:
            motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
            print(f"\n[FAULT] IMU out of range (roll={roll:.1f} pitch={pitch:.1f}) — stopped.")
            break

        eff_roll  = roll  if abs(roll)  > ROLL_DEADBAND_DEG  else 0.0
        eff_pitch = pitch if abs(pitch) > PITCH_DEADBAND_DEG else 0.0

        d_roll  = max(-MAX_D_ERPM, min(ROLL_KD  * filt_roll_rate,  MAX_D_ERPM))
        d_pitch = max(-MAX_D_ERPM, min(PITCH_KD * filt_pitch_rate, MAX_D_ERPM))

        # Update IMU2 baseline with slow EMA to track DC drift (tau ~20s).
        # This keeps FF centred as the boat's static tilt slowly shifts.
        if imu2_available:
            imu2_roll_baseline  = (1 - IMU2_BASELINE_ALPHA) * imu2_roll_baseline  + IMU2_BASELINE_ALPHA * imu2_roll
            imu2_pitch_baseline = (1 - IMU2_BASELINE_ALPHA) * imu2_pitch_baseline + IMU2_BASELINE_ALPHA * imu2_pitch

        # Feedforward: pre-correct using boat motion (IMU2) before platform tilts.
        # Subtract running baseline so FF only reacts to changes, not DC offset.
        ff_roll  = ROLL_KFF  * (imu2_roll  - imu2_roll_baseline)  if imu2_available else 0.0
        ff_pitch = PITCH_KFF * (imu2_pitch - imu2_pitch_baseline) if imu2_available else 0.0

        cmd_roll  = max(-MAX_ERPM, min(SIGN_ROLL  * (ROLL_GAIN  * eff_roll  + d_roll  + ff_roll),  MAX_ERPM))
        cmd_pitch = max(-MAX_ERPM, min(SIGN_PITCH * (PITCH_GAIN * eff_pitch + d_pitch + ff_pitch), MAX_ERPM))


        if roll_ok:  motor_roll.set_rpm(cmd_roll)
        if pitch_ok: motor_pitch.set_rpm(cmd_pitch)

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

        if motor_roll_amps > MAX_MOTOR_AMPS or motor_pitch_amps > MAX_MOTOR_AMPS:
            overcurrent_cycles += 1
            if overcurrent_cycles >= MAX_OVERCURRENT_CYCLES:
                motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
                print(f"\n[FAULT] Overcurrent — roll:{motor_roll_amps:.2f}A pitch:{motor_pitch_amps:.2f}A — stopped.")
                break
        else:
            overcurrent_cycles = 0

        imu2_str = f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}" if imu2_available else "IMU2:N/A"
        print(
            f"t={loop_count*LOOP_PERIOD:07.2f}s | "
            f"Roll:{roll:+6.2f}=>{cmd_roll:+5.0f}ERPM {motor_roll_amps:.2f}A | "
            f"Pitch:{pitch:+6.2f}=>{cmd_pitch:+5.0f}ERPM {motor_pitch_amps:.2f}A | "
            f"{imu2_str} | dt:{dt*1000:.1f}ms",
            end="\r",
        )

        log_csv.writerow([
            f"{loop_count*LOOP_PERIOD:.3f}",
            f"{roll:.4f}",    f"{cmd_roll:.1f}",
            f"{pitch:.4f}",   f"{cmd_pitch:.1f}",
            f"{imu2_roll:.4f}", f"{imu2_pitch:.4f}",
            f"{motor_roll_erpm:.0f}", f"{motor_pitch_erpm:.0f}",
            f"{motor_roll_pos:.2f}",  f"{motor_pitch_pos:.2f}",
            f"{motor_roll_amps:.3f}", f"{motor_pitch_amps:.3f}",
            f"{filt_roll_rate:.2f}", f"{filt_pitch_rate:.2f}",
            f"{d_roll:.1f}",  f"{d_pitch:.1f}",
            f"{ff_roll:.1f}", f"{ff_pitch:.1f}",
            f"{dt*1000:.2f}",
        ])

        loop_count += 1
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
