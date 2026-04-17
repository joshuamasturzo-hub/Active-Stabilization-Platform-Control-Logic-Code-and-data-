"""Safeguard calibration script.

Motors are held at zero torque — completely free to rotate manually.
Rotate the platform to each physical extreme and hold for a few seconds.
The script logs IMU angle and motor tachometer position the whole time.
After Ctrl+C the CSV shows the peak values at each limit, which are then
used to set the safeguards in main_vel_control.py.

Procedure:
    1. Run this script
    2. Slowly rotate platform to MAX roll in one direction, hold 3s
    3. Return to center
    4. Slowly rotate to MAX roll in other direction, hold 3s
    5. Return to center
    6. Repeat steps 2-5 for pitch
    7. Ctrl+C
    8. Read the printed limit summary

Usage:
    python3 calibration.py
"""

import csv
import time

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"
LOOP_PERIOD: float = 0.02

print("[INFO] Initialising IMU1 (bus 7)...")
imu1 = IMUReader(i2c_bus=7)
print("[INFO] IMU1 ready.")

motor_roll  = SerialMotorDriver(port=PORT_ROLL,  motor_id=1)
motor_pitch = SerialMotorDriver(port=PORT_PITCH, motor_id=2)

roll_ok  = motor_roll.connect()
pitch_ok = motor_pitch.connect()

# Arm at zero torque — motors are on but apply NO force
if roll_ok:
    motor_roll.arm()
    print(f"[INFO] Roll  motor armed (zero torque) on {PORT_ROLL}")
if pitch_ok:
    motor_pitch.arm()
    print(f"[INFO] Pitch motor armed (zero torque) on {PORT_PITCH}")

# Read tachometer starting positions as offset baseline
_roll_offset = _pitch_offset = 0.0
if roll_ok:
    motor_roll.request_telemetry(); time.sleep(0.1)
    s = motor_roll.get_state()
    if s: _roll_offset = s["pos"]
if pitch_ok:
    motor_pitch.request_telemetry(); time.sleep(0.1)
    s = motor_pitch.get_state()
    if s: _pitch_offset = s["pos"]

print(f"[INFO] Tachometer offsets — roll:{_roll_offset:+.1f}  pitch:{_pitch_offset:+.1f}")
print()
print("[INFO] Motors are FREE TO ROTATE — zero torque applied.")
print("[INFO] Slowly rotate the platform to each physical limit and hold.")
print("[INFO] Press Ctrl+C when done.")
print()

LOG_FILE = "calibration_data.csv"
log_fh  = open(LOG_FILE, "w", newline="")
log_csv = csv.writer(log_fh)
log_csv.writerow(["t_s", "imu_roll_deg", "imu_pitch_deg",
                  "motor_roll_pos_deg", "motor_pitch_pos_deg"])

motor_roll_pos  = 0.0
motor_pitch_pos = 0.0

# Tracking peaks
peak_roll_pos_max  = 0.0
peak_roll_pos_min  = 0.0
peak_pitch_pos_max = 0.0
peak_pitch_pos_min = 0.0
peak_imu_roll_max  = 0.0
peak_imu_roll_min  = 0.0
peak_imu_pitch_max = 0.0
peak_imu_pitch_min = 0.0

try:
    loop_count = 0
    while True:
        loop_start = time.perf_counter()

        # Keep sending zero torque every cycle so motors stay free
        if roll_ok:
            motor_roll.send_torque(0.0)
        if pitch_ok:
            motor_pitch.send_torque(0.0)

        # Read IMU
        angles = imu1.get_angles()
        roll   = angles["roll"]
        pitch  = angles["pitch"]

        # Request and read telemetry
        if roll_ok:
            motor_roll.request_telemetry()
        if pitch_ok:
            motor_pitch.request_telemetry()

        if roll_ok:
            s = motor_roll.get_state()
            if s: motor_roll_pos = s["pos"] - _roll_offset
        if pitch_ok:
            s = motor_pitch.get_state()
            if s: motor_pitch_pos = s["pos"] - _pitch_offset

        # Track peaks
        peak_roll_pos_max  = max(peak_roll_pos_max,  motor_roll_pos)
        peak_roll_pos_min  = min(peak_roll_pos_min,  motor_roll_pos)
        peak_pitch_pos_max = max(peak_pitch_pos_max, motor_pitch_pos)
        peak_pitch_pos_min = min(peak_pitch_pos_min, motor_pitch_pos)
        peak_imu_roll_max  = max(peak_imu_roll_max,  roll)
        peak_imu_roll_min  = min(peak_imu_roll_min,  roll)
        peak_imu_pitch_max = max(peak_imu_pitch_max, pitch)
        peak_imu_pitch_min = min(peak_imu_pitch_min, pitch)

        print(
            f"IMU  roll:{roll:+6.2f} deg  pitch:{pitch:+6.2f} deg  |  "
            f"Motor roll:{motor_roll_pos:+7.1f} deg  pitch:{motor_pitch_pos:+7.1f} deg",
            end="\r"
        )

        log_csv.writerow([
            f"{loop_count * LOOP_PERIOD:.3f}",
            f"{roll:.4f}", f"{pitch:.4f}",
            f"{motor_roll_pos:.2f}", f"{motor_pitch_pos:.2f}",
        ])

        loop_count += 1
        used = time.perf_counter() - loop_start
        sleep_time = LOOP_PERIOD - used
        if sleep_time > 0.0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n")
    print("=" * 55)
    print("  CALIBRATION RESULTS — copy these into main_vel_control.py")
    print("=" * 55)
    print(f"  IMU roll  range:  {peak_imu_roll_min:+.2f} to {peak_imu_roll_max:+.2f} deg")
    print(f"  IMU pitch range:  {peak_imu_pitch_min:+.2f} to {peak_imu_pitch_max:+.2f} deg")
    print(f"  Motor roll  range: {peak_roll_pos_min:+.1f} to {peak_roll_pos_max:+.1f} deg")
    print(f"  Motor pitch range: {peak_pitch_pos_min:+.1f} to {peak_pitch_pos_max:+.1f} deg")
    print()
    print("  Recommended safeguards (90% of measured limits):")
    print(f"  IMU_ROLL_LIMIT  = {0.9 * max(abs(peak_imu_roll_min), abs(peak_imu_roll_max)):.1f} deg")
    print(f"  IMU_PITCH_LIMIT = {0.9 * max(abs(peak_imu_pitch_min), abs(peak_imu_pitch_max)):.1f} deg")
    print(f"  MOTOR_ROLL_LIMIT  = {0.9 * max(abs(peak_roll_pos_min), abs(peak_roll_pos_max)):.1f} deg")
    print(f"  MOTOR_PITCH_LIMIT = {0.9 * max(abs(peak_pitch_pos_min), abs(peak_pitch_pos_max)):.1f} deg")
    print("=" * 55)

finally:
    motor_roll.stop()
    motor_pitch.stop()
    log_fh.close()
    print(f"[INFO] Data saved to {LOG_FILE}")
