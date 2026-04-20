# Active Stabilisation Platform — FPD Headless Tune

Two-axis active stabilisation platform for drone landing at sea. A Jetson runs a **200 Hz FF+PD control loop** over USB-to-UART serial, using two BNO08x IMUs — one on the platform (feedback) and one on the boat hull (feedforward) — to drive two CubeMars AK60-39 V3.0 actuators on the roll and pitch axes.

---

## Quick Start

```bash
ssh edg5@172.20.10.10
cd ~/captone_OG
python3 main_fpd_tune_headless.py
```

The script creates `gains.json` with defaults on first run. To tune live, edit `gains.json` in a second terminal while it's running — changes are picked up every 0.5 s without restarting.

```bash
# In a second SSH terminal
nano gains.json
```

Press `Ctrl+C` to stop. Motors are zeroed and ports closed automatically. A timestamped CSV and analysis PNG are saved to the working directory.

---

## Hardware Setup

### Wiring

| Connection | Details |
|---|---|
| Roll motor | CubeMars AK60-39 V3.0 → `/dev/ttyROLL` |
| Pitch motor | CubeMars AK60-39 V3.0 → `/dev/ttyPITCH` |
| IMU1 (platform) | BNO08x → Jetson I2C bus 7 |
| IMU2 (boat hull) | BNO08x → Jetson I2C bus 1 |
| Motor power | 22 V supply |

**Motor direction:** Roll sign is −1 (inverted), Pitch sign is +1. Set via `SIGN_ROLL` and `SIGN_PITCH` in `main_fpd_tune_headless.py`.

### USB Port Enumeration

`/dev/ttyROLL` and `/dev/ttyPITCH` are symlinked to the correct `/dev/ttyUSBx` ports. If a port isn't found, verify with:

```bash
ls /dev/ttyUSB*
dmesg | grep ttyUSB | tail -5
```

### SSH into the Jetson

```bash
ssh edg5@172.20.10.10
```

---

## Software Architecture

### `code/main_fpd_tune_headless.py` — Main Control Loop

Runs a **200 Hz FF+PD control loop** headless (no display required). Designed for SSH operation.

**Control law:**
```
erpm = SIGN * (Kp * imu1_angle + Ki * integral + Kd * imu1_rate + Kff * imu2_rate)
```

| Term | Description |
|---|---|
| `Kp * imu1_angle` | P — reacts to current platform tilt |
| `Ki * integral` | I — eliminates steady-state error |
| `Kd * imu1_rate` | D — damps platform's own motion (IMU1 gyro) |
| `Kff * imu2_rate` | FF — anticipates boat disturbance (IMU2 gyro) |

**Control mode:** ERPM (velocity loop on the motor's internal controller). The Jetson sends ERPM commands; the motor handles current internally.

**Live gain reload:** Every 100 loops (~0.5 s) the script re-reads `gains.json`. Any changed value takes effect immediately. A `[GAIN CHANGE]` line is printed and logged. Delete `gains.json` to reset all gains to defaults.

**Auto-plot:** After each run a timestamped `_analysis.png` is saved alongside the CSV — 8-panel analysis including roll/pitch angles, commands, current, quarter-segment RMS stats, and FFT.

### `code/imu_sensor.py` — IMU HAL

BNO08x over I2C (ExtendedI2C). Returns `{"roll", "pitch", "yaw"}` in degrees and `{"roll_rate", "pitch_rate"}` in deg/s. Falls back to last good reading on I2C errors; reinitialises after 5 consecutive failures. Retries feature enable up to 10× on startup (BNO08x needs up to 5 s after cold power-on).

### `code/serial_motor_driver.py` — UART Motor HAL

One instance per motor. CubeMars native serial protocol (AK series v3.2.0), 921600 8N1.

| Method | Description |
|---|---|
| `connect()` | Open serial port |
| `arm()` | Send 10× zero-ERPM frames to wake controller |
| `set_erpm(erpm)` | Send `COMM_SET_RPM` (0x49) ERPM command |
| `request_telemetry()` | Send `COMM_GET_VALUES` (0x45) request |
| `get_state()` | Drain RX buffer, return latest `{erpm, amps, voltage}` |
| `stop()` | Zero ERPM and close port |

**Frame format:** `[0xAA][Len][Cmd][Data...][CRC16_H][CRC16_L][0xBB]`
CRC16-CCITT: poly `0x1021`, init `0x0000`, MSB-first, no reflection.

### `code/motor_test.py` — Pre-Run Utility

```bash
python3 code/motor_test.py recover    # drive pitch back to 0° using IMU feedback
python3 code/motor_test.py sweep      # controlled ±10° sweep to verify response
```

### `code/system_check_uart.py` — Pre-Flight Diagnostics

```bash
python3 code/system_check_uart.py
```

Checks USB-UART ports, motor response, IMU, and I2C buses before a run.

---

## Gain Tuning

### gains.json — Tunable Parameters

| Parameter | Default | Description |
|---|---|---|
| `ROLL_GAIN` | 600.0 | Roll proportional gain (ERPM/deg) |
| `PITCH_GAIN` | 300.0 | Pitch proportional gain (ERPM/deg) |
| `ROLL_KI` | 3.0 | Roll integral gain |
| `PITCH_KI` | 2.0 | Pitch integral gain |
| `MAX_I_ERPM` | 300.0 | Integral anti-windup clamp (ERPM) |
| `ROLL_KD` | 10.0 | Roll derivative gain |
| `PITCH_KD` | −5.0 | Pitch derivative gain (negative = inverted axis) |
| `MAX_D_ERPM` | 600.0 | Derivative output clamp (ERPM) |
| `D_FILTER_ALPHA` | 0.15 | D-term second-order EMA filter (0=heavy, 1=none) |
| `ROLL_KFF` | 70.0 | Roll feedforward gain (ERPM per deg/s of IMU2) |
| `PITCH_KFF` | 40.0 | Pitch feedforward gain |
| `FF_FILTER_ALPHA` | 0.4 | FF-term EMA filter |
| `FF_RATE_DEADBAND_DPS` | 1.5 | Zero FF below this IMU2 rate (prevents noise amp) |
| `ANGLE_FILTER_ALPHA` | 1.0 | Platform angle EMA filter (1.0 = off) |
| `ROLL_DEADBAND_DEG` | 0.5 | Roll deadband — no command below this angle |
| `PITCH_DEADBAND_DEG` | 1.0 | Pitch deadband |
| `MAX_MOTOR_AMPS` | 10.0 | Motor overcurrent trip threshold (A) |
| `MAX_ERPM` | 5000.0 | Total command clamp (ERPM) |

### Known Stability Limits

| Axis | Gain | Notes |
|---|---|---|
| Roll | 600 stable | 750+ oscillates without sufficient D damping |
| Pitch | 300 stable | 400+ oscillates without sufficient D damping |

### Tuning Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Oscillation at all times | Kp too high | Reduce GAIN |
| Slow drift, never settles | Kp too low | Increase GAIN |
| Overshoots then oscillates | Kd too low | Increase KD |
| Jerky, high-frequency noise | Kd too high or D filter too open | Reduce KD or lower D_FILTER_ALPHA |
| Steady-state offset | KI too low | Increase KI |
| FF causes oscillation at rest | FF_RATE_DEADBAND too low | Increase FF_RATE_DEADBAND_DPS |

---

## Data Collection

Each run saves two files timestamped at start:

- `fpd_tune_data_YYYYMMDD_HHMMSS.csv` — full log
- `fpd_tune_data_YYYYMMDD_HHMMSS_analysis.png` — auto-generated 8-panel plot

### CSV Columns

| Column | Unit | Description |
|---|---|---|
| `t_s` | s | Elapsed time |
| `imu1_roll_deg` | deg | Platform roll (IMU1) |
| `imu1_pitch_deg` | deg | Platform pitch (IMU1) |
| `imu2_roll_deg` | deg | Boat roll reference (IMU2) |
| `imu2_pitch_deg` | deg | Boat pitch reference (IMU2) |
| `imu1_roll_rate_dps` | deg/s | Platform roll rate |
| `imu1_pitch_rate_dps` | deg/s | Platform pitch rate |
| `imu2_roll_rate_dps` | deg/s | Boat roll rate (FF source) |
| `imu2_pitch_rate_dps` | deg/s | Boat pitch rate (FF source) |
| `i_roll_erpm` | ERPM | Integral contribution — roll |
| `i_pitch_erpm` | ERPM | Integral contribution — pitch |
| `d_roll_erpm` | ERPM | Derivative contribution — roll |
| `d_pitch_erpm` | ERPM | Derivative contribution — pitch |
| `ff_roll_erpm` | ERPM | Feedforward contribution — roll |
| `ff_pitch_erpm` | ERPM | Feedforward contribution — pitch |
| `cmd_roll_erpm` | ERPM | Total roll command sent to motor |
| `cmd_pitch_erpm` | ERPM | Total pitch command sent to motor |
| `motor_roll_erpm` | ERPM | Measured roll motor speed (telemetry) |
| `motor_pitch_erpm` | ERPM | Measured pitch motor speed (telemetry) |
| `motor_roll_amps` | A | Measured roll motor current (telemetry) |
| `motor_pitch_amps` | A | Measured pitch motor current (telemetry) |
| `dt_ms` | ms | Actual loop period (target: 5 ms / 200 Hz) |
| `gain_event` | string | Gain change description if gains reloaded this loop |

---

## Hardware Parameter Map

| Parameter | Value |
|---|---|
| Roll motor port | `/dev/ttyROLL` |
| Pitch motor port | `/dev/ttyPITCH` |
| Baud rate | 921,600 bps, 8N1 |
| Control loop frequency | 200 Hz (5 ms period) |
| IMU1 (platform) | BNO08x, I2C bus 7 |
| IMU2 (boat hull) | BNO08x, I2C bus 1 |
| Motor supply voltage | 22 V |
| Roll motor direction | Inverted (`SIGN_ROLL = −1.0`) |
| Pitch motor direction | Normal (`SIGN_PITCH = +1.0`) |

---

## Communication Protocol Reference

### CubeMars Serial Protocol (AK Series v3.2.0)

| Command | ID | Description |
|---|---|---|
| `COMM_GET_VALUES` | `0x45` | Request full telemetry |
| `COMM_SET_CURRENT` | `0x47` | Set phase current (torque mode) |
| `COMM_SET_RPM` | `0x49` | Set ERPM (velocity loop) ← used by this script |
| `COMM_SET_POS` | `0x4A` | Position loop (motor internal PID) |

### Telemetry Response (`COMM_GET_VALUES`)

| Offset | Type | Scale | Field |
|---|---|---|---|
| 0–1 | int16 | ÷10 | MOS temperature (°C) |
| 2–3 | int16 | ÷10 | Motor temperature (°C) |
| 4–7 | int32 | ÷100 | Output current (A) |
| 8–11 | int32 | ÷100 | Input current (A) |
| 20–21 | int16 | ÷1000 | Duty cycle |
| 22–25 | int32 | raw | Motor speed (ERPM) |
| 26–27 | int16 | ÷10 | Input voltage (V) |
