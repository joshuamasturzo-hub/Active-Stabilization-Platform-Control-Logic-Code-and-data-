"""Velocity-loop stabilization for the marine drone landing platform.

Why velocity control
--------------------
Position control needs a known reference: a specific motor angle that means
"platform is level."  On a boat that powers up at any angle, that reference
is never known without a calibration routine.

Velocity control has no such requirement:
  - Command 0 ERPM when IMU reads 0 deg  => motor holds wherever it is
  - Command negative ERPM when IMU reads +10 deg => motor spins to correct
  - Command positive ERPM when IMU reads -10 deg => motor spins to correct
  - Works identically from any starting angle

Control law
-----------
    erpm_cmd = -GAIN * imu_angle

The motor's own velocity PID handles the dynamics; we only set the setpoint.
A proportional gain (GAIN) maps degrees of tilt to ERPM.

Tuning guide
------------
  GAIN too low  -> motor barely responds, platform drifts
  GAIN too high -> overshoot, oscillation (increase DEADBAND or lower GAIN)
  MAX_ERPM      -> safety cap, limits how aggressively the motor corrects

Start with the defaults.  If the platform oscillates at rest, halve GAIN.
If it doesn't correct fast enough, double GAIN.

Signs
-----
SIGN_ROLL = -1.0, SIGN_PITCH = +1.0 from torque-mode convention.
If a motor corrects in the wrong direction, flip its sign.

Hardware
--------
  IMU1:  BNO08x on I2C bus 7, address 0x4A  (primary -- drives control)
  IMU2:  BNO08x on I2C bus 1, address 0x4A  (reference -- log only)
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_vel_control.py
"""

import csv
import sys
import time

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

# ── Port Configuration ────────────────────────────────────────────────────────

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"

# ── Control Parameters ────────────────────────────────────────────────────────

LOOP_PERIOD: float = 0.02   # 50 Hz

# ERPM per degree of tilt.  Start conservative.
# AK60-39: 1000 ERPM = 47.6 shaft RPM = 286 deg/s shaft speed.
ROLL_GAIN:  float = 600.0   # ERPM per degree of tilt
PITCH_GAIN: float = 300.0

# Motor direction signs.  Flip to -1.0 if the motor corrects the wrong way.
SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

# Hard ERPM cap.  AK60-39 rated max output ~70 RPM. 5000 ERPM is well within safe range.
MAX_ERPM: float = 5000.0

# Deadbands: tilts smaller than this are ignored (avoids hunting at level).
ROLL_DEADBAND_DEG:  float = 0.5
PITCH_DEADBAND_DEG: float = 1.0


# Current protection — AK60-39 rated 4.5 A continuous.
# If either motor exceeds this for MAX_OVERCURRENT_CYCLES consecutive cycles,
# motors are stopped and the script exits.
MAX_MOTOR_AMPS:       float = 4.0   # A  (leave headroom below 4.5 A rated)
MAX_OVERCURRENT_CYCLES: int = 3     # ~60 ms at 50 Hz before trip




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

# ── IMU Stabilisation Wait ────────────────────────────────────────────────────

print("[INFO] Waiting 3 s for IMU to stabilise...")
for i in range(3, 0, -1):
    a = imu1.get_angles()
    print(f"  {i}s  roll:{a['roll']:+6.2f} deg  pitch:{a['pitch']:+6.2f} deg", end="\r")
    time.sleep(1.0)
print("\n[INFO] Ready.")
print(f"[INFO] P-only | ROLL_GAIN={ROLL_GAIN}  PITCH_GAIN={PITCH_GAIN}  MAX_ERPM={MAX_ERPM}  DEADBAND roll={ROLL_DEADBAND_DEG} pitch={PITCH_DEADBAND_DEG} deg")
print(f"[INFO] SIGN_ROLL={SIGN_ROLL}  SIGN_PITCH={SIGN_PITCH}")

# ── Logging Setup ─────────────────────────────────────────────────────────────

import datetime
LOG_FILE = f"vel_control_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
    "dt_ms",
])
print(f"[INFO] Logging to {LOG_FILE}")
print("[INFO] Entering 50 Hz control loop.  Ctrl+C to stop.")

# ── 50 Hz Control Loop ────────────────────────────────────────────────────────

try:
    # Read starting motor positions so the ±30° limit is always relative
    # to THIS run's start, regardless of accumulated tachometer history.
    _roll_offset  = 0.0
    _pitch_offset = 0.0
    if roll_ok:
        motor_roll.request_telemetry()
        time.sleep(0.1)
        s = motor_roll.get_state()
        if s:
            _roll_offset = s["pos"]
    if pitch_ok:
        motor_pitch.request_telemetry()
        time.sleep(0.1)
        s = motor_pitch.get_state()
        if s:
            _pitch_offset = s["pos"]
    print(f"[INFO] Motor position offsets — roll:{_roll_offset:+.1f} deg  pitch:{_pitch_offset:+.1f} deg")

    prev_time: float = time.perf_counter()
    loop_count: int  = 0
    motor_roll_erpm:  float = 0.0
    motor_pitch_erpm: float = 0.0
    motor_roll_pos:   float = 0.0   # position relative to this run's start
    motor_pitch_pos:  float = 0.0
    motor_roll_amps:  float = 0.0
    motor_pitch_amps: float = 0.0
    overcurrent_cycles: int = 0

    while True:
        loop_start: float = time.perf_counter()
        dt: float = loop_start - prev_time
        prev_time = loop_start

        # ── IMU Read ─────────────────────────────────────────────────
        angles1 = imu1.get_angles()
        roll    = angles1["roll"]
        pitch   = angles1["pitch"]

        imu2_roll = imu2_pitch = 0.0
        if imu2_available:
            angles2    = imu2.get_angles()
            imu2_roll  = angles2["roll"]
            imu2_pitch = angles2["pitch"]

        # ── IMU Sanity Check ──────────────────────────────────────────
        # If IMU returns a physically impossible angle, stop motors.
        if abs(roll) > 60.0 or abs(pitch) > 60.0:
            motor_roll.set_rpm(0)
            motor_pitch.set_rpm(0)
            print(f"\n[FAULT] IMU reading out of range (roll={roll:.1f} pitch={pitch:.1f}) — motors stopped.")
            break

        # ── Velocity Command ──────────────────────────────────────────
        eff_roll  = roll  if abs(roll)  > ROLL_DEADBAND_DEG  else 0.0
        eff_pitch = pitch if abs(pitch) > PITCH_DEADBAND_DEG else 0.0

        raw_erpm_roll  = SIGN_ROLL  * ROLL_GAIN  * eff_roll
        raw_erpm_pitch = SIGN_PITCH * PITCH_GAIN * eff_pitch

        cmd_roll  = max(-MAX_ERPM, min(raw_erpm_roll,  MAX_ERPM))
        cmd_pitch = max(-MAX_ERPM, min(raw_erpm_pitch, MAX_ERPM))


        # ── Send Commands ─────────────────────────────────────────────
        if roll_ok:
            motor_roll.set_rpm(cmd_roll)
        if pitch_ok:
            motor_pitch.set_rpm(cmd_pitch)

        # ── Request Telemetry ─────────────────────────────────────────
        if roll_ok:
            motor_roll.request_telemetry()
        if pitch_ok:
            motor_pitch.request_telemetry()

        # ── Read Feedback ─────────────────────────────────────────────
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
                motor_roll.set_rpm(0)
                motor_pitch.set_rpm(0)
                print(f"\n[FAULT] Overcurrent — roll:{motor_roll_amps:.2f}A  pitch:{motor_pitch_amps:.2f}A  (limit {MAX_MOTOR_AMPS}A) — motors stopped.")
                break
        else:
            overcurrent_cycles = 0

        # ── Terminal ──────────────────────────────────────────────────
        imu2_str = (f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}"
                    if imu2_available else "IMU2:N/A")
        print(
            f"t={loop_count * LOOP_PERIOD:07.2f}s | "
            f"Roll:{roll:+6.2f}=>{cmd_roll:+5.0f}ERPM {motor_roll_amps:.2f}A | "
            f"Pitch:{pitch:+6.2f}=>{cmd_pitch:+5.0f}ERPM {motor_pitch_amps:.2f}A | "
            f"{imu2_str} | dt:{dt * 1000:.1f}ms",
            end="\r",
        )

        # ── Log ───────────────────────────────────────────────────────
        log_csv.writerow([
            f"{loop_count * LOOP_PERIOD:.3f}",
            f"{roll:.4f}",       f"{cmd_roll:.1f}",
            f"{pitch:.4f}",      f"{cmd_pitch:.1f}",
            f"{imu2_roll:.4f}",  f"{imu2_pitch:.4f}",
            f"{motor_roll_erpm:.0f}",
            f"{motor_pitch_erpm:.0f}",
            f"{motor_roll_pos:.2f}",
            f"{motor_pitch_pos:.2f}",
            f"{motor_roll_amps:.3f}",
            f"{motor_pitch_amps:.3f}",
            f"{dt * 1000:.2f}",
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
    if roll_ok:
        motor_roll.set_rpm(0)
    if pitch_ok:
        motor_pitch.set_rpm(0)
    time.sleep(0.3)
    motor_roll.stop()
    motor_pitch.stop()
    log_fh.close()
    print(f"[INFO] Stopped. Data saved to {LOG_FILE}.")
