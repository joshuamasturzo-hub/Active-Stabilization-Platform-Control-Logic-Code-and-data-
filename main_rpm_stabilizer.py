"""Stabilization loop — RPM derivative edition.

Control architecture (per axis):
    cmd = Kp × angle_error  −  Kd × motor_shaft_deg_per_s

    P-term : IMU1 angle error           — low-frequency, noisy → proportional only
    D-term : motor ERPM → shaft deg/s   — high-frequency, clean → velocity damping

    IMU2 (bus 1) is logged as a cross-reference and never drives the controller.

Startup runs a full hardware self-test before arming any motors:
    • IMU1 and IMU2 — I2C communication, live data, angle sanity
    • Roll motor    — UART link, bus voltage, temperature, RPM at rest, fault code
    • Pitch motor   — same

Hardware:
    IMU1  : BNO08x on I2C bus 7  (primary — stabilization)
    IMU2  : BNO08x on I2C bus 1  (reference — log only)
    Roll  : CubeMars AK60-39 V3.0 on /dev/ttyROLL
    Pitch : CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_rpm_stabilizer.py
"""

import csv
import sys
import time

from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

# ── Port configuration ────────────────────────────────────────────────────────

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"

# ── Motor constants ───────────────────────────────────────────────────────────

POLE_PAIRS: int = 21                          # AK60-39 V3.0
ERPM_TO_DEG_S: float = 6.0 / POLE_PAIRS      # ERPM → mechanical deg/s (≈ 0.286)

# ── Control parameters ────────────────────────────────────────────────────────

LOOP_PERIOD:  float = 0.02    # 50 Hz
TARGET_ANGLE: float = 0.0     # level-hold setpoint (degrees)

# Proportional gain — same units as before (A / deg)
ROLL_KP:  float = 0.35
PITCH_KP: float = 0.35

# Derivative gain — applied to shaft deg/s  (A / (deg/s))
ROLL_KD:  float = 0.003
PITCH_KD: float = 0.003

MAX_TORQUE: float = 3.0   # A — hard output clamp

# Deadband — ignore angle errors smaller than this.
# Platform must actually be displaced before motors activate.
DEADBAND_DEG: float = 1.5

# Hard position limit — motors cut if they travel more than this from home.
POSITION_LIMIT_DEG: float = 45.0

# RPM sign: +1 if positive ERPM → platform angle increases, -1 if inverted.
# If the D-term makes oscillation WORSE instead of damping it, flip the sign.
ROLL_RPM_SIGN:  float = +1.0
PITCH_RPM_SIGN: float = +1.0

# ── Self-test thresholds ──────────────────────────────────────────────────────

MIN_VOLTAGE:      float = 15.0   # V  — fail below this
WARN_VOLTAGE:     float = 22.0   # V  — warn below this (full charge ~24 V)
MAX_TEMP_FAULT:   float = 80.0   # °C — fail above this
WARN_TEMP:        float = 60.0   # °C — warn above this
MAX_IDLE_RPM:     int   = 50     # ERPM — warn if spinning at rest
POSITION_LIMIT:   float = 360.0  # deg — safety trip-wire from start position


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'='*54}")
    print(f"  {title}")
    print(f"{'='*54}")


def _check(label: str, ok: bool, value: str = "", warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    icon   = "✓" if ok else ("!" if warn else "✗")
    print(f"    [{icon}] {label:<28} {value:<16}  {status}")


def test_imu(label: str, bus: int) -> tuple[bool, object]:
    """Initialise an IMU, read 20 samples live, verify it is alive and sane.

    Returns (passed: bool, imu_instance_or_None).
    """
    print(f"\n  {label} (I2C bus {bus})")

    try:
        imu = IMUReader(i2c_bus=bus)
    except Exception as exc:
        print(f"    [✗] Initialisation FAILED: {exc}")
        return False, None

    _check("Initialised", True, f"bus {bus}")

    print(f"    Reading 20 samples (live):")
    samples = []
    for i in range(20):
        a = imu.get_angles()
        samples.append(a)
        print(f"      [{i+1:02d}/20]  roll={a['roll']:+7.3f}°  "
              f"pitch={a['pitch']:+7.3f}°  yaw={a['yaw']:+8.3f}°",
              end="\r")
        time.sleep(0.05)
    print()

    rolls   = [s["roll"]  for s in samples]
    pitches = [s["pitch"] for s in samples]

    roll_range  = max(rolls)  - min(rolls)
    pitch_range = max(pitches) - min(pitches)
    alive      = (roll_range > 0.0 or pitch_range > 0.0)
    mean_roll  = sum(rolls)  / len(rolls)
    mean_pitch = sum(pitches) / len(pitches)
    sane       = abs(mean_roll) < 180.0 and abs(mean_pitch) < 90.0

    _check("Data live (not frozen)", alive,
           f"Δroll={roll_range:.3f}° Δpitch={pitch_range:.3f}°")
    _check("Angles in range", sane,
           f"roll={mean_roll:+.2f}° pitch={mean_pitch:+.2f}°")
    _check("Pitch reasonable (<45°)", abs(mean_pitch) < 45.0,
           f"{mean_pitch:+.2f}°", warn=abs(mean_pitch) >= 45.0)

    passed = alive and sane
    print(f"  → {'PASS' if passed else 'FAIL'}")
    return passed, imu


def test_motor(label: str, port: str, motor_id: int) -> tuple[bool, object]:
    """Open motor UART, request telemetry, validate power and state.

    Returns (passed: bool, motor_instance_or_None).
    """
    print(f"\n  {label} ({port})")
    motor = SerialMotorDriver(port=port, motor_id=motor_id)

    if not motor.connect():
        print(f"    [✗] Serial open FAILED — is {port} plugged in?")
        return False, None
    _check("Serial link opened", True, port)

    # Request telemetry three times with a short pause — use last valid reply
    state = None
    for attempt in range(3):
        motor.request_telemetry()
        time.sleep(0.05)
        s = motor.get_state()
        if s is not None:
            state = s

    if state is None:
        print("    [✗] No telemetry response — motor powered?")
        motor.stop()
        return False, None
    _check("Telemetry received", True)

    voltage  = state["voltage"]
    temp     = state["temp_fet"]
    erpm     = abs(state["rpm"])
    current  = abs(state["torque"])
    fault    = state["fault"]

    volt_ok  = voltage >= MIN_VOLTAGE
    volt_warn= voltage < WARN_VOLTAGE
    temp_ok  = temp < MAX_TEMP_FAULT
    temp_warn= temp >= WARN_TEMP
    rpm_ok   = True          # just warn if spinning
    rpm_warn = erpm > MAX_IDLE_RPM
    curr_ok  = current < 2.0
    fault_ok = fault == 0

    _check("Bus voltage",    volt_ok,  f"{voltage:.1f} V",
           warn=volt_ok and volt_warn)
    _check("FET temperature", temp_ok, f"{temp:.1f} °C",
           warn=temp_ok and temp_warn)
    _check("Shaft at rest",  not rpm_warn, f"{erpm} ERPM",
           warn=rpm_warn)
    _check("Current ~0 A",   curr_ok,  f"{current:.2f} A")
    _check("Fault code",     fault_ok, f"0x{fault:02X} ({'NONE' if fault==0 else 'ACTIVE'})")

    passed = volt_ok and temp_ok and curr_ok and fault_ok
    print(f"  → {'PASS' if passed else 'FAIL'}")
    return passed, motor


def run_self_test() -> dict:
    """Run all hardware checks. Returns dict of initialised hardware or exits."""
    _banner("HARDWARE SELF-TEST")

    print("\n── IMUs ──────────────────────────────────────────────")
    imu1_ok, imu1 = test_imu("IMU1 (stabilisation)", bus=7)

    # IMU2 — non-critical, same graceful fallback as main_dual_imu.py
    print(f"\n  IMU2 (reference) (I2C bus 1)")
    try:
        imu2    = IMUReader(i2c_bus=1)
        imu2_ok = True
        print(f"    [✓] Initialised                  bus 1             PASS")
        # show a quick live feed
        print(f"    Reading 10 samples (live):")
        for i in range(10):
            a = imu2.get_angles()
            print(f"      [{i+1:02d}/10]  roll={a['roll']:+7.3f}°  "
                  f"pitch={a['pitch']:+7.3f}°  yaw={a['yaw']:+8.3f}°", end="\r")
            time.sleep(0.05)
        print()
        print(f"  → PASS")
    except Exception as exc:
        imu2    = None
        imu2_ok = False
        print(f"    [✗] Initialisation FAILED: {exc}")
        print(f"  → WARN (non-critical — imu2 columns will be 0.0)")

    print("\n── Motors ────────────────────────────────────────────")
    roll_ok,  motor_roll  = test_motor("Roll  motor", PORT_ROLL,  motor_id=1)
    pitch_ok, motor_pitch = test_motor("Pitch motor", PORT_PITCH, motor_id=2)

    _banner("SELF-TEST SUMMARY")
    results = {
        "IMU1 (stabilisation)": imu1_ok,
        "IMU2 (reference)":     imu2_ok,
        "Roll  motor":          roll_ok,
        "Pitch motor":          pitch_ok,
    }
    all_critical_ok = imu1_ok and roll_ok and pitch_ok

    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {name}")

    if not all_critical_ok:
        print("\n[FATAL] One or more critical components failed. Fix issues and retry.")
        # Clean up anything that opened
        for m in (motor_roll, motor_pitch):
            if m is not None:
                m.stop()
        sys.exit(1)

    if not imu2_ok:
        print("\n[WARN] IMU2 unavailable — imu2 CSV columns will be 0.0.")

    print("\nAll critical checks passed.")

    # ── Live IMU monitor ──────────────────────────────────────────────────────
    # Show both IMUs live for 10 s so you can verify readings before arming.
    # Press Ctrl+C here to abort without touching motors.
    print("\n── Live IMU Feed (10 s) — verify readings before arming ─────────────")
    print(f"  {'':4}  {'IMU1 (stabilisation)':30}  {'IMU2 (reference)':30}")
    print(f"  {'':4}  {'roll':>10}  {'pitch':>10}  {'yaw':>10}  {'roll':>10}  {'pitch':>10}")
    try:
        for i in range(500):   # 10 s at 50 ms each
            a1 = imu1.get_angles()
            a2 = imu2.get_angles() if imu2_ok else {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            elapsed = i * 0.02
            print(
                f"  {elapsed:5.1f}s"
                f"  {a1['roll']:>+10.3f}  {a1['pitch']:>+10.3f}  {a1['yaw']:>+10.3f}"
                f"  {a2['roll']:>+10.3f}  {a2['pitch']:>+10.3f}",
                end="\r",
            )
            time.sleep(0.02)
        print()
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        for m in (motor_roll, motor_pitch):
            if m is not None:
                m.stop()
        sys.exit(0)

    print("\n─────────────────────────────────────────────────────────────────────")
    try:
        input("Press ENTER to arm motors and begin stabilisation (Ctrl+C to abort): ")
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        for m in (motor_roll, motor_pitch):
            if m is not None:
                m.stop()
        sys.exit(0)

    return {
        "imu1":        imu1,
        "imu2":        imu2,
        "imu2_ok":     imu2_ok,
        "motor_roll":  motor_roll,
        "motor_pitch": motor_pitch,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    hw = run_self_test()

    imu1        = hw["imu1"]
    imu2        = hw["imu2"]
    imu2_ok     = hw["imu2_ok"]
    motor_roll  = hw["motor_roll"]
    motor_pitch = hw["motor_pitch"]

    # ── IMU stabilisation wait ────────────────────────────────────────────────
    print("\n[INFO] Waiting 3 s for IMUs to stabilise...")
    for i in range(3, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1.0)
    print("[INFO] IMUs ready.          ")

    # ── Arm motors ────────────────────────────────────────────────────────────
    motor_roll.arm()
    motor_pitch.arm()
    print("[INFO] Motors armed.")

    # ── Record start positions for safety trip-wire ───────────────────────────
    motor_roll.request_telemetry();  time.sleep(0.1)
    motor_pitch.request_telemetry(); time.sleep(0.1)
    _rs = motor_roll.get_state()
    _ps = motor_pitch.get_state()
    roll_pos_start  = _rs["pos"] if _rs else 0.0
    pitch_pos_start = _ps["pos"] if _ps else 0.0
    print(f"[INFO] Start positions — Roll: {roll_pos_start:.1f}°  Pitch: {pitch_pos_start:.1f}°")

    # ── CSV log ───────────────────────────────────────────────────────────────
    LOG_FILE = "rpm_stabilizer_data.csv"
    log_fh   = open(LOG_FILE, "w", newline="")
    log_csv  = csv.writer(log_fh)
    log_csv.writerow([
        "t_s",
        "roll_deg",   "roll_erpm",   "cmd_roll_A",
        "pitch_deg",  "pitch_erpm",  "cmd_pitch_A",
        "imu2_roll_deg", "imu2_pitch_deg",
        "roll_volt",  "pitch_volt",
        "dt_ms",
    ])
    print(f"[INFO] Logging to {LOG_FILE}")
    print(f"[INFO] Entering control loop at {1/LOOP_PERIOD:.0f} Hz. Ctrl+C to stop.\n")

    # ── Motor state cache (updated each cycle from previous telemetry reply) ──
    roll_erpm   = 0
    pitch_erpm  = 0
    roll_volt   = 0.0
    pitch_volt  = 0.0
    roll_pos    = roll_pos_start
    pitch_pos   = pitch_pos_start

    imu2_roll  = 0.0
    imu2_pitch = 0.0

    try:
        prev_time:  float = time.perf_counter()
        loop_count: int   = 0

        while True:
            loop_start: float = time.perf_counter()
            dt: float = loop_start - prev_time
            prev_time = loop_start

            # ── IMU reads ─────────────────────────────────────────────────
            angles = imu1.get_angles()

            if imu2_ok:
                a2 = imu2.get_angles()
                imu2_roll  = a2["roll"]
                imu2_pitch = a2["pitch"]

            # ── Deadband — do nothing if platform is essentially level ────
            error_roll  = TARGET_ANGLE - angles["roll"]
            error_pitch = TARGET_ANGLE - angles["pitch"]

            if abs(error_roll)  < DEADBAND_DEG:
                error_roll  = 0.0
            if abs(error_pitch) < DEADBAND_DEG:
                error_pitch = 0.0

            # ── RPM-based D-term ──────────────────────────────────────────
            roll_vel_deg_s  = roll_erpm  * ERPM_TO_DEG_S * ROLL_RPM_SIGN
            pitch_vel_deg_s = pitch_erpm * ERPM_TO_DEG_S * PITCH_RPM_SIGN

            raw_roll  = ROLL_KP  * error_roll  - ROLL_KD  * roll_vel_deg_s
            raw_pitch = PITCH_KP * error_pitch - PITCH_KD * pitch_vel_deg_s

            cmd_roll  = max(-MAX_TORQUE, min(raw_roll,  MAX_TORQUE))
            cmd_pitch = max(-MAX_TORQUE, min(raw_pitch, MAX_TORQUE))

            # ── 45° hard position limit ───────────────────────────────────
            if abs(roll_pos  - roll_pos_start)  > POSITION_LIMIT_DEG:
                print(f"\n[SAFETY] Roll  exceeded ±{POSITION_LIMIT_DEG:.0f}° — axis disabled.")
                cmd_roll = 0.0
            if abs(pitch_pos - pitch_pos_start) > POSITION_LIMIT_DEG:
                print(f"\n[SAFETY] Pitch exceeded ±{POSITION_LIMIT_DEG:.0f}° — axis disabled.")
                cmd_pitch = 0.0

            # ── Motor output ──────────────────────────────────────────────
            motor_roll.send_torque(cmd_roll)
            motor_pitch.send_torque(-cmd_pitch)   # pitch motor physically inverted

            # ── Request telemetry (reply arrives next cycle) ──────────────
            motor_roll.request_telemetry()
            motor_pitch.request_telemetry()

            # ── Parse telemetry from previous cycle ───────────────────────
            sr = motor_roll.get_state()
            if sr is not None:
                roll_erpm  = sr["rpm"]
                roll_pos   = sr["pos"]
                roll_volt  = sr["voltage"]

            sp = motor_pitch.get_state()
            if sp is not None:
                pitch_erpm  = sp["rpm"]
                pitch_pos   = sp["pos"]
                pitch_volt  = sp["voltage"]

            # ── Terminal status ───────────────────────────────────────────
            imu2_str = (
                f"IMU2 R:{imu2_roll:+6.2f}° P:{imu2_pitch:+6.2f}°"
                if imu2_ok else "IMU2:N/A          "
            )
            print(
                f"t={loop_count * LOOP_PERIOD:07.2f}s | "
                f"Roll:{angles['roll']:+7.2f}° {roll_erpm:+6d}rpm cmd:{cmd_roll:+6.3f}A | "
                f"Pitch:{angles['pitch']:+7.2f}° {pitch_erpm:+6d}rpm cmd:{cmd_pitch:+6.3f}A | "
                f"{imu2_str} | dt:{dt*1000:5.1f}ms",
                end="\r",
            )

            # ── CSV log ───────────────────────────────────────────────────
            log_csv.writerow([
                f"{loop_count * LOOP_PERIOD:.3f}",
                f"{angles['roll']:.4f}",   f"{roll_erpm}",   f"{cmd_roll:.4f}",
                f"{angles['pitch']:.4f}",  f"{pitch_erpm}",  f"{cmd_pitch:.4f}",
                f"{imu2_roll:.4f}",        f"{imu2_pitch:.4f}",
                f"{roll_volt:.2f}",        f"{pitch_volt:.2f}",
                f"{dt * 1000:.2f}",
            ])

            loop_count += 1

            # ── Rate limiting ─────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
