"""FF+PD TUNING SCRIPT — live gain adjustment via gains.json.

Control law
-----------
    erpm = SIGN * (Kp * imu1_angle + Kd * imu1_rate + Kff * imu2_rate)

  - P term  (Kp * imu1_angle):  reacts to current tilt of platform
  - D term  (Kd * imu1_rate):   damps platform's own motion (IMU1 gyro)
  - FF term (Kff * imu2_rate):  anticipates boat disturbance (IMU2 gyro)

Live tuning
-----------
  Edit gains.json while this script is running. Changes are picked up
  every 0.5 s (100 loops) without restarting.  A [GAIN CHANGE] line is
  printed and logged whenever a value changes.

  To reset to defaults, delete gains.json — it will be recreated.

  Tunable fields in gains.json:
    ROLL_GAIN, PITCH_GAIN
    ROLL_KD, PITCH_KD
    ROLL_KFF, PITCH_KFF
    MAX_D_ERPM
    D_FILTER_ALPHA, FF_FILTER_ALPHA (second-order EMA — steeper rolloff than single EMA)
    FF_RATE_DEADBAND_DPS  (zero FF below this IMU2 rate — prevents noise amplification at rest)
    ROLL_DEADBAND_DEG, PITCH_DEADBAND_DEG
    MAX_MOTOR_AMPS, MAX_ERPM

Known stability ceilings (from testing):
  PITCH_GAIN: 300 stable, 400+ oscillates without sufficient D damping
  ROLL_GAIN:  600 stable, 750+ oscillates

Live plot
---------
  Set LIVE_PLOT = True to show a real-time scrolling graph while running.
  Requires a display (monitor or X11 forwarding).
  Set LIVE_PLOT = False for headless / SSH without X.

Hardware
--------
  IMU1: BNO08x on I2C bus 7  (platform — P + D feedback)
  IMU2: BNO08x on I2C bus 1  (boat reference — feedforward)
  Roll:  CubeMars AK60-39 V3.0 on /dev/ttyROLL
  Pitch: CubeMars AK60-39 V3.0 on /dev/ttyPITCH

Usage:
    python3 main_fpd_tune.py
    # In another terminal while it's running:
    nano gains.json
"""

import collections
import csv
import datetime
import json
import os
import sys
import threading
import time

# ── Live plot toggle ──────────────────────────────────────────────────────────
# Set False for headless / SSH without X forwarding.
LIVE_PLOT: bool = False

# Scrolling window shown in the live plot (seconds).
LIVE_WINDOW_S: float = 12.0


def generate_plot(csv_file: str, gains_snapshot: dict) -> None:
    """Auto-generate analysis plot after each run — same format as manual plots."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.fft import rfft, rfftfreq

        df = pd.read_csv(csv_file)
        if len(df) < 50:
            print("[PLOT] Too few samples — skipping plot.")
            return

        t  = df["t_s"].values
        r1 = df["imu1_roll_deg"].values;  p1 = df["imu1_pitch_deg"].values
        r2 = df["imu2_roll_deg"].values;  p2 = df["imu2_pitch_deg"].values
        cr = df["cmd_roll_erpm"].values;  cp = df["cmd_pitch_erpm"].values
        dr = df["d_roll_erpm"].values;    dp = df["d_pitch_erpm"].values
        fr = df["ff_roll_erpm"].values;   fp = df["ff_pitch_erpm"].values
        ra = df["motor_roll_amps"].values; pa = df["motor_pitch_amps"].values

        n = len(t); dt_med = np.median(np.diff(t))
        freqs = rfftfreq(n, d=dt_med)
        qs = [0, n//4, n//2, 3*n//4, n-1]
        qt = [t[i] for i in qs]
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        qc     = ['#FFF9C4', '#E8F5E9', '#FFF3E0', '#F3E5F5']

        def dom_f(s):
            m = np.abs(rfft(s - s.mean()))
            return freqs[np.argmax(m[1:]) + 1]

        def quarter_stats(ref, plat):
            out = []
            for i in range(4):
                sl = slice(qs[i], qs[i+1])
                rr = np.sqrt(np.mean(ref[sl]**2))
                rp = np.sqrt(np.mean(plat[sl]**2))
                out.append((rr, rp, (1 - rp/rr)*100 if rr > 0 else 0))
            return out

        def quarter_period(sig):
            out = []
            for i in range(4):
                sl = slice(qs[i], qs[i+1]); s = sig[sl]
                f = rfftfreq(len(s), d=dt_med)
                m = np.abs(rfft(s - s.mean()))
                pf = f[np.argmax(m[1:]) + 1] if len(m) > 1 else 0.1
                out.append(1/pf if pf > 0 else 0)
            return out

        ir = df["i_roll_erpm"].values  if "i_roll_erpm"  in df.columns else np.zeros(n)
        ip = df["i_pitch_erpm"].values if "i_pitch_erpm" in df.columns else np.zeros(n)
        pr = cr - ir - dr - fr
        pp = cp - ip - dp - fp

        rms_r2 = np.sqrt(np.mean(r2**2)); rms_r1 = np.sqrt(np.mean(r1**2))
        rms_p2 = np.sqrt(np.mean(p2**2)); rms_p1 = np.sqrt(np.mean(p1**2))
        red_r = (1 - rms_r1/rms_r2)*100; red_p = (1 - rms_p1/rms_p2)*100
        r_stats = quarter_stats(r2, r1)
        p_stats = quarter_stats(p2, p1)
        roll_T  = quarter_period(r2)
        pitch_T = quarter_period(p2)

        gtxt = (f"PITCH_GAIN={gains_snapshot.get('PITCH_GAIN','?')} "
                f"PITCH_KD={gains_snapshot.get('PITCH_KD','?')} "
                f"ROLL_KFF={gains_snapshot.get('ROLL_KFF','?')} "
                f"PITCH_KFF={gains_snapshot.get('PITCH_KFF','?')} "
                f"MAX_D={gains_snapshot.get('MAX_D_ERPM','?')}")

        fig = plt.figure(figsize=(16, 18)); fig.patch.set_facecolor('#f8f9fa')
        gs2 = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)
        fig.suptitle(f"FF+PD TUNE — {os.path.basename(csv_file)}\n"
                     f"{gtxt}  |  Roll {red_r:.1f}%  Pitch {red_p:.1f}%",
                     fontsize=11, fontweight='bold', y=0.99)

        for col, (ref, plat, stats, T, name, db, red) in enumerate([
            (r2, r1, r_stats, roll_T,  "Roll",  0.5, red_r),
            (p2, p1, p_stats, pitch_T, "Pitch", 1.0, red_p),
        ]):
            ax = fig.add_subplot(gs2[0, col])
            for i in range(4):
                ax.axvspan(qt[i], qt[i+1], alpha=0.15, color=qc[i], zorder=0)
            ax.plot(t, ref,  'b-', alpha=0.6, lw=0.9, label='IMU2 (ref)')
            ax.plot(t, plat, 'r-', alpha=0.85, lw=1.1, label='Platform (IMU1)')
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.fill_between(t, -db, db, alpha=0.08, color='green', label=f'±{db}° deadband')
            ymax = max(np.abs(ref).max(), np.abs(plat).max()) * 1.15 or 1.0
            for i in range(4):
                mid = (qt[i] + qt[i+1]) / 2
                ax.text(mid,  ymax*0.92, f"{stats[i][2]:.0f}%", ha='center',
                        fontsize=9, color=colors[i], fontweight='bold')
                ax.text(mid, -ymax*0.88, f"T={T[i]:.1f}s", ha='center', fontsize=7, color='gray')
            ax.set_ylim(-ymax, ymax)
            ax.set_title(f"{name} Angle  |  Overall {red:.1f}% reduction", fontsize=10, fontweight='bold')
            ax.set_ylabel("Degrees"); ax.set_xlabel("Time (s)")
            ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.25)
            ax.set_xlim(t[0], t[-1])

        for col, (p_t, i_t, ff_t, d_t, cmd, name) in enumerate([
            (pr, ir, fr, dr, cr, "Roll"), (pp, ip, fp, dp, cp, "Pitch")
        ]):
            ax = fig.add_subplot(gs2[1, col])
            ax.fill_between(t, 0, p_t,  alpha=0.35, color='#1976D2', label=f'P (max={np.abs(p_t).max():.0f})')
            ax.fill_between(t, 0, i_t,  alpha=0.35, color='#E91E63', label=f'I (max={np.abs(i_t).max():.0f})')
            ax.fill_between(t, 0, ff_t, alpha=0.35, color='#388E3C', label=f'FF (max={np.abs(ff_t).max():.0f})')
            ax.fill_between(t, 0, d_t,  alpha=0.35, color='#F57C00', label=f'D (max={np.abs(d_t).max():.0f})')
            ax.plot(t, cmd, 'k-', lw=0.8, alpha=0.7, label='Total cmd')
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_title(f"{name} Command  (P / I / FF / D breakdown)", fontsize=9, fontweight='bold')
            ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
            ax.legend(fontsize=7, loc='upper right', ncol=2); ax.grid(True, alpha=0.25)
            ax.set_xlim(t[0], t[-1])

        ax = fig.add_subplot(gs2[2, 0])
        ax.plot(t, ra, 'b-', lw=0.9, alpha=0.85, label='Roll')
        ax.plot(t, pa, 'r-', lw=0.9, alpha=0.85, label='Pitch')
        ax.axhline(4.5, color='red', lw=1.0, ls=':', label='4.5A rated')
        ax.set_title(f"Motor Current  |  Roll max={ra.max():.2f}A  Pitch max={pa.max():.2f}A",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("Amps"); ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

        ax = fig.add_subplot(gs2[2, 1])
        cap = gains_snapshot.get('MAX_D_ERPM', 600)
        ax.plot(t, np.abs(dr), 'b-', lw=0.9, alpha=0.85, label=f'D roll (max={np.abs(dr).max():.0f})')
        ax.plot(t, np.abs(dp), 'r-', lw=0.9, alpha=0.85, label=f'D pitch (max={np.abs(dp).max():.0f})')
        ax.axhline(cap, color='k', lw=0.8, ls=':', label=f'D cap {cap:.0f}')
        ax.set_title(f"|D Term| Magnitude  (max {max(np.abs(dr).max(), np.abs(dp).max()):.0f} ERPM)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

        ax = fig.add_subplot(gs2[3, :])
        x = np.arange(4); w = 0.35
        r_reds = [s[2] for s in r_stats]; p_reds = [s[2] for s in p_stats]
        b1 = ax.bar(x-w/2, r_reds, w, label='Roll',  color='#1976D2', alpha=0.85)
        b2 = ax.bar(x+w/2, p_reds, w, label='Pitch', color='#D32F2F', alpha=0.85)
        for bars, c in [(b1,'#1976D2'), (b2,'#D32F2F')]:
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f'{bar.get_height():.0f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=c)
        xlabels = [f'Q{i+1}\nT≈{roll_T[i]:.1f}s' for i in range(4)]
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("% Reduction")
        ax.set_ylim(min(0, min(r_reds+p_reds))-10, 110)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title("Stabilization by Phase  |  Wave period (T) shown below each bar",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='y')

        out_file = csv_file.replace(".csv", "_analysis.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Saved: {out_file}")

        # Pop open the image automatically
        import subprocess
        try:
            subprocess.Popen(["xdg-open", out_file])
        except Exception:
            pass

    except Exception as e:
        print(f"[PLOT] Failed to generate plot: {e}")


from imu_sensor import IMUReader
from serial_motor_driver import SerialMotorDriver

# ── Port Configuration ────────────────────────────────────────────────────────

PORT_ROLL:  str = "/dev/ttyROLL"
PORT_PITCH: str = "/dev/ttyPITCH"

LOOP_PERIOD: float = 0.005   # 200 Hz
GAINS_FILE:  str   = "gains.json"
GAINS_RELOAD_EVERY: int = 100  # loops (~0.5 s)

# ── Default gains (written to gains.json on first run) ───────────────────────

DEFAULTS = {
    "ROLL_GAIN":           600.0,
    "PITCH_GAIN":          300.0,
    "ROLL_KI":             3.0,
    "PITCH_KI":            2.0,
    "MAX_I_ERPM":          300.0,
    "ROLL_KD":             10.0,
    "PITCH_KD":            -5.0,
    "MAX_D_ERPM":          600.0,
    "D_FILTER_ALPHA":      0.15,
    "ROLL_KFF":            70.0,
    "PITCH_KFF":           40.0,
    "FF_FILTER_ALPHA":     0.4,
    "FF_RATE_DEADBAND_DPS": 1.5,
    "ANGLE_FILTER_ALPHA":  1.0,
    "ROLL_DEADBAND_DEG":   0.5,
    "PITCH_DEADBAND_DEG":  1.0,
    "MAX_MOTOR_AMPS":      10.0,
    "MAX_ERPM":            5000.0,
}

SIGN_ROLL:  float = -1.0
SIGN_PITCH: float = +1.0

MAX_OVERCURRENT_CYCLES: int = 15   # ~75 ms at 200 Hz

# ── Shared live-plot buffer ───────────────────────────────────────────────────
# Deque appends are thread-safe in CPython (GIL).
# The plot thread only reads snapshots — no locks needed.

_LIVE_MAX = int(LIVE_WINDOW_S / LOOP_PERIOD)
_buf: dict = {k: collections.deque(maxlen=_LIVE_MAX) for k in
              ('t', 'r1', 'p1', 'r2', 'p2', 'cr', 'cp', 'ra', 'pa', 'dr', 'dp')}
_ctrl_stop  = threading.Event()   # set to stop the control thread
_ctrl_fault = threading.Event()   # set when control thread hits a fault


def load_gains(path: str, current: dict) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        merged = dict(current)
        for k in DEFAULTS:
            if k in data:
                merged[k] = float(data[k])
        return merged
    except Exception as e:
        print(f"\n[WARN] gains.json read error ({e}) — keeping current gains")
        return current


def save_defaults(path: str):
    with open(path, "w") as f:
        json.dump(DEFAULTS, f, indent=2)
    print(f"[INFO] Created {path} with default gains.")


def gains_diff_str(old: dict, new: dict) -> str:
    changes = []
    for k in DEFAULTS:
        if abs(old.get(k, 0) - new.get(k, 0)) > 1e-9:
            changes.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    return ", ".join(changes)


# ── Write defaults if gains.json doesn't exist ───────────────────────────────

if not os.path.exists(GAINS_FILE):
    save_defaults(GAINS_FILE)

gains = load_gains(GAINS_FILE, dict(DEFAULTS))

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
print()
print(f"[INFO] FF+PD TUNE | ROLL  Kp={gains['ROLL_GAIN']} Kd={gains['ROLL_KD']}  Kff={gains['ROLL_KFF']}")
print(f"[INFO]            | PITCH Kp={gains['PITCH_GAIN']} Kd={gains['PITCH_KD']} Kff={gains['PITCH_KFF']}")
print(f"[INFO] MAX_ERPM={gains['MAX_ERPM']} | MAX_D_ERPM={gains['MAX_D_ERPM']} | MAX_AMPS={gains['MAX_MOTOR_AMPS']}")
print(f"[INFO] Live tuning: edit {GAINS_FILE} while running — reloaded every {GAINS_RELOAD_EVERY} loops")
if not imu2_available:
    print("[WARN] Running PD-only (no IMU2 for feedforward).")

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_FILE = f"fpd_tune_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_fh   = open(LOG_FILE, "w", newline="")
log_csv  = csv.writer(log_fh)
log_csv.writerow([
    "t_s",
    "imu1_roll_deg", "imu1_pitch_deg",
    "imu2_roll_deg", "imu2_pitch_deg",
    "imu1_roll_rate_dps", "imu1_pitch_rate_dps",
    "imu2_roll_rate_dps", "imu2_pitch_rate_dps",
    "i_roll_erpm", "i_pitch_erpm",
    "d_roll_erpm", "d_pitch_erpm",
    "ff_roll_erpm", "ff_pitch_erpm",
    "cmd_roll_erpm", "cmd_pitch_erpm",
    "motor_roll_erpm", "motor_pitch_erpm",
    "motor_roll_amps", "motor_pitch_amps",
    "dt_ms",
    "gain_event",
])
print(f"[INFO] Logging to {LOG_FILE}")
print("[INFO] Entering 200 Hz control loop.  Ctrl+C to stop.")
if LIVE_PLOT:
    print("[INFO] Live plot window opening — close it or press Ctrl+C to stop.")

# ── Control loop (runs in background thread when LIVE_PLOT=True) ──────────────

def control_loop():
    global gains

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
    roll_integral  = 0.0   # accumulated angle error (deg·s)
    pitch_integral = 0.0
    filt_imu1_roll_rate  = 0.0
    filt_imu1_pitch_rate = 0.0
    filt_imu2_roll_rate  = 0.0
    filt_imu2_pitch_rate = 0.0
    filt2_imu2_roll_rate  = 0.0
    filt2_imu2_pitch_rate = 0.0
    filt_roll  = 0.0
    filt_pitch = 0.0
    overcurrent_cycles: int = 0
    gain_event: str = ""

    try:
        while not _ctrl_stop.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - prev_time
            prev_time = loop_start
            gain_event = ""

            # ── Reload gains periodically ─────────────────────────────
            if loop_count > 0 and loop_count % GAINS_RELOAD_EVERY == 0:
                new_gains = load_gains(GAINS_FILE, gains)
                diff = gains_diff_str(gains, new_gains)
                if diff:
                    gain_event = f"CHANGED: {diff}"
                    print(f"\n[GAIN CHANGE] t={loop_count*LOOP_PERIOD:.2f}s | {diff}")
                    if abs(new_gains["D_FILTER_ALPHA"] - gains["D_FILTER_ALPHA"]) > 1e-9:
                        filt_imu1_roll_rate = filt_imu1_pitch_rate = 0.0
                    if abs(new_gains["FF_FILTER_ALPHA"] - gains["FF_FILTER_ALPHA"]) > 1e-9:
                        filt_imu2_roll_rate = filt_imu2_pitch_rate = 0.0
                        filt2_imu2_roll_rate = filt2_imu2_pitch_rate = 0.0
                    if abs(new_gains["ANGLE_FILTER_ALPHA"] - gains["ANGLE_FILTER_ALPHA"]) > 1e-9:
                        filt_roll = filt_pitch = 0.0
                if abs(new_gains["ROLL_KI"] - gains["ROLL_KI"]) > 1e-9 or \
                   abs(new_gains["PITCH_KI"] - gains["PITCH_KI"]) > 1e-9:
                        roll_integral = pitch_integral = 0.0
                        print("\n[INFO] KI changed — integrals reset to 0.")
                gains = new_gains

            # ── Unpack gains ──────────────────────────────────────────
            ROLL_GAIN             = gains["ROLL_GAIN"]
            PITCH_GAIN            = gains["PITCH_GAIN"]
            ROLL_KI               = gains["ROLL_KI"]
            PITCH_KI              = gains["PITCH_KI"]
            MAX_I_ERPM            = gains["MAX_I_ERPM"]
            ROLL_KD               = gains["ROLL_KD"]
            PITCH_KD              = gains["PITCH_KD"]
            MAX_D_ERPM            = gains["MAX_D_ERPM"]
            D_FILTER_ALPHA        = gains["D_FILTER_ALPHA"]
            ROLL_KFF              = gains["ROLL_KFF"]
            PITCH_KFF             = gains["PITCH_KFF"]
            FF_FILTER_ALPHA       = gains["FF_FILTER_ALPHA"]
            FF_RATE_DEADBAND_DPS  = gains["FF_RATE_DEADBAND_DPS"]
            ROLL_DEADBAND_DEG     = gains["ROLL_DEADBAND_DEG"]
            PITCH_DEADBAND_DEG    = gains["PITCH_DEADBAND_DEG"]
            MAX_MOTOR_AMPS        = gains["MAX_MOTOR_AMPS"]
            MAX_ERPM              = gains["MAX_ERPM"]
            ANGLE_FILTER_ALPHA    = gains["ANGLE_FILTER_ALPHA"]

            # ── IMU1 ──────────────────────────────────────────────────
            angles1 = imu1.get_angles()
            raw_roll  = angles1["roll"]
            raw_pitch = angles1["pitch"]
            filt_roll  = ANGLE_FILTER_ALPHA * raw_roll  + (1 - ANGLE_FILTER_ALPHA) * filt_roll
            filt_pitch = ANGLE_FILTER_ALPHA * raw_pitch + (1 - ANGLE_FILTER_ALPHA) * filt_pitch
            roll  = filt_roll
            pitch = filt_pitch
            rates1 = imu1.get_rates()
            imu1_roll_rate  = rates1["roll_rate"]
            imu1_pitch_rate = rates1["pitch_rate"]
            filt_imu1_roll_rate  = D_FILTER_ALPHA * imu1_roll_rate  + (1 - D_FILTER_ALPHA) * filt_imu1_roll_rate
            filt_imu1_pitch_rate = D_FILTER_ALPHA * imu1_pitch_rate + (1 - D_FILTER_ALPHA) * filt_imu1_pitch_rate

            # ── IMU2 ──────────────────────────────────────────────────
            imu2_roll = imu2_pitch = 0.0
            imu2_roll_rate = imu2_pitch_rate = 0.0
            if imu2_available:
                angles2 = imu2.get_angles()
                imu2_roll  = angles2["roll"]
                imu2_pitch = angles2["pitch"]
                rates2 = imu2.get_rates()
                imu2_roll_rate  = rates2["roll_rate"]
                imu2_pitch_rate = rates2["pitch_rate"]
                filt_imu2_roll_rate   = FF_FILTER_ALPHA * imu2_roll_rate        + (1 - FF_FILTER_ALPHA) * filt_imu2_roll_rate
                filt2_imu2_roll_rate  = FF_FILTER_ALPHA * filt_imu2_roll_rate   + (1 - FF_FILTER_ALPHA) * filt2_imu2_roll_rate
                filt_imu2_pitch_rate  = FF_FILTER_ALPHA * imu2_pitch_rate       + (1 - FF_FILTER_ALPHA) * filt_imu2_pitch_rate
                filt2_imu2_pitch_rate = FF_FILTER_ALPHA * filt_imu2_pitch_rate  + (1 - FF_FILTER_ALPHA) * filt2_imu2_pitch_rate

            # ── IMU Sanity ────────────────────────────────────────────
            if abs(roll) > 60.0 or abs(pitch) > 60.0:
                motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
                print(f"\n[FAULT] IMU out of range (roll={roll:.1f} pitch={pitch:.1f}) — stopped.")
                _ctrl_fault.set()
                break

            # ── Control Law ───────────────────────────────────────────
            eff_roll  = roll  if abs(roll)  > ROLL_DEADBAND_DEG  else 0.0
            eff_pitch = pitch if abs(pitch) > PITCH_DEADBAND_DEG else 0.0

            # ── Integral (accumulate only outside deadband) ───────────
            roll_integral  += eff_roll  * dt
            pitch_integral += eff_pitch * dt
            # Anti-windup: clamp so I contribution never exceeds MAX_I_ERPM
            if ROLL_KI  > 0: roll_integral  = max(-MAX_I_ERPM/ROLL_KI,  min(roll_integral,  MAX_I_ERPM/ROLL_KI))
            if PITCH_KI > 0: pitch_integral = max(-MAX_I_ERPM/PITCH_KI, min(pitch_integral, MAX_I_ERPM/PITCH_KI))
            i_roll  = ROLL_KI  * roll_integral
            i_pitch = PITCH_KI * pitch_integral

            d_roll  = max(-MAX_D_ERPM, min(ROLL_KD  * filt_imu1_roll_rate,  MAX_D_ERPM))
            d_pitch = max(-MAX_D_ERPM, min(PITCH_KD * filt_imu1_pitch_rate, MAX_D_ERPM))

            ff_roll_rate  = filt2_imu2_roll_rate  if abs(filt2_imu2_roll_rate)  > FF_RATE_DEADBAND_DPS else 0.0
            ff_pitch_rate = filt2_imu2_pitch_rate if abs(filt2_imu2_pitch_rate) > FF_RATE_DEADBAND_DPS else 0.0
            ff_roll  = ROLL_KFF  * ff_roll_rate
            ff_pitch = PITCH_KFF * ff_pitch_rate

            cmd_roll  = max(-MAX_ERPM, min(SIGN_ROLL  * (ROLL_GAIN * eff_roll  + i_roll  + d_roll  + ff_roll),  MAX_ERPM))
            cmd_pitch = max(-MAX_ERPM, min(SIGN_PITCH * (PITCH_GAIN * eff_pitch + i_pitch + d_pitch + ff_pitch), MAX_ERPM))

            # ── Send ──────────────────────────────────────────────────
            if roll_ok:  motor_roll.set_rpm(cmd_roll)
            if pitch_ok: motor_pitch.set_rpm(cmd_pitch)

            # ── Telemetry ─────────────────────────────────────────────
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

            # ── Current protection ────────────────────────────────────
            if motor_roll_amps > MAX_MOTOR_AMPS or motor_pitch_amps > MAX_MOTOR_AMPS:
                overcurrent_cycles += 1
                if overcurrent_cycles >= MAX_OVERCURRENT_CYCLES:
                    motor_roll.set_rpm(0); motor_pitch.set_rpm(0)
                    print(f"\n[FAULT] Overcurrent — roll:{motor_roll_amps:.2f}A pitch:{motor_pitch_amps:.2f}A — stopped.")
                    _ctrl_fault.set()
                    break
            else:
                overcurrent_cycles = 0

            # ── Push to live buffer ───────────────────────────────────
            _t = loop_count * LOOP_PERIOD
            _buf['t'].append(_t);   _buf['r1'].append(roll);    _buf['p1'].append(pitch)
            _buf['r2'].append(imu2_roll); _buf['p2'].append(imu2_pitch)
            _buf['cr'].append(cmd_roll);  _buf['cp'].append(cmd_pitch)
            _buf['ra'].append(motor_roll_amps); _buf['pa'].append(motor_pitch_amps)
            _buf['dr'].append(d_roll);    _buf['dp'].append(d_pitch)

            # ── Terminal (every 20 loops) ─────────────────────────────
            if loop_count % 20 == 0:
                imu2_str = f"IMU2 R:{imu2_roll:+6.2f} P:{imu2_pitch:+6.2f}" if imu2_available else "IMU2:N/A"
                print(
                    f"t={_t:07.2f}s | "
                    f"Roll:{roll:+6.2f}=>{cmd_roll:+5.0f}ERPM(I:{i_roll:+4.0f} D:{d_roll:+4.0f}) {motor_roll_amps:.2f}A | "
                    f"Pitch:{pitch:+6.2f}=>{cmd_pitch:+5.0f}ERPM(I:{i_pitch:+4.0f} D:{d_pitch:+4.0f}) {motor_pitch_amps:.2f}A | "
                    f"{imu2_str} | dt:{dt*1000:.1f}ms",
                    end="\r",
                )

            # ── Log ───────────────────────────────────────────────────
            log_csv.writerow([
                f"{_t:.4f}",
                f"{roll:.4f}",      f"{pitch:.4f}",
                f"{imu2_roll:.4f}", f"{imu2_pitch:.4f}",
                f"{filt_imu1_roll_rate:.3f}", f"{filt_imu1_pitch_rate:.3f}",
                f"{filt_imu2_roll_rate:.3f}", f"{filt_imu2_pitch_rate:.3f}",
                f"{i_roll:.1f}",    f"{i_pitch:.1f}",
                f"{d_roll:.1f}",    f"{d_pitch:.1f}",
                f"{ff_roll:.1f}",   f"{ff_pitch:.1f}",
                f"{cmd_roll:.1f}",  f"{cmd_pitch:.1f}",
                f"{motor_roll_erpm:.0f}", f"{motor_pitch_erpm:.0f}",
                f"{motor_roll_amps:.3f}", f"{motor_pitch_amps:.3f}",
                f"{dt*1000:.2f}",
                gain_event,
            ])

            loop_count += 1
            used = time.perf_counter() - loop_start
            sleep_time = LOOP_PERIOD - used
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Control loop stopping — zeroing motors...")
        if roll_ok:  motor_roll.set_rpm(0)
        if pitch_ok: motor_pitch.set_rpm(0)
        time.sleep(0.3)
        motor_roll.stop()
        motor_pitch.stop()
        log_fh.close()
        print(f"[INFO] Data saved to {LOG_FILE}.")
        generate_plot(LOG_FILE, gains)
        _ctrl_stop.set()


# ── Entry point ───────────────────────────────────────────────────────────────

if not LIVE_PLOT:
    import signal
    signal.signal(signal.SIGINT, lambda s, f: _ctrl_stop.set())
    control_loop()
else:
    # Matplotlib must be in the main thread.
    # Control loop runs in a daemon background thread.
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    import signal

    def _sigint(sig, frame):
        print("\n[INFO] Ctrl+C — stopping...")
        _ctrl_stop.set()
        plt.close('all')

    signal.signal(signal.SIGINT, _sigint)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # Per-cycle period history (one point per detected wave cycle)
    _period_buf_t    = collections.deque(maxlen=300)
    _period_buf_roll = collections.deque(maxlen=300)   # None = no roll crossing this entry
    _period_buf_ptch = collections.deque(maxlen=300)   # None = no pitch crossing this entry
    # Zero-crossing state: last crossing time and last processed timestamp
    _zc_state = {'r_t_last': None, 'p_t_last': None, 'last_t': -1.0}

    # ── Build live figure (5 panels) ──────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#1a1a2e')
    # rows: roll, pitch get 3 units height; commands, amps, period get 2 units
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 3, 2, 2, 2], hspace=0.55)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    fig.suptitle("FF+PD Live Monitor  —  edit gains.json to tune  —  Ctrl+C or close to stop",
                 color='white', fontsize=11, fontweight='bold')

    # Pre-create line objects
    l_r2,  = axes[0].plot([], [], 'b-',  alpha=0.6, lw=1.0, label='IMU2 ref')
    l_r1,  = axes[0].plot([], [], color='#FF6B6B', lw=1.2, label='IMU1 platform')
    l_p2,  = axes[1].plot([], [], 'b-',  alpha=0.6, lw=1.0, label='IMU2 ref')
    l_p1,  = axes[1].plot([], [], color='#FF6B6B', lw=1.2, label='IMU1 platform')
    l_cr,  = axes[2].plot([], [], color='#4CAF50', lw=1.0, label='Roll cmd')
    l_cp,  = axes[2].plot([], [], color='#CE93D8', lw=1.0, label='Pitch cmd')
    l_ra,  = axes[3].plot([], [], color='#4CAF50', lw=1.0, label='Roll amps')
    l_pa,  = axes[3].plot([], [], color='#CE93D8', lw=1.0, label='Pitch amps')
    axes[3].axhline(4.5, color='red', lw=0.8, ls=':', label='4.5A rated')
    l_pr,  = axes[4].plot([], [], color='#4FC3F7', lw=0, marker='o', ms=4, label='Roll period')
    l_pp,  = axes[4].plot([], [], color='#FFB74D', lw=0, marker='s', ms=4, label='Pitch period')

    labels = [
        ('Roll (°)',  'Roll  —  blue=reference  red=platform'),
        ('Pitch (°)', 'Pitch —  blue=reference  red=platform'),
        ('ERPM',      'Motor commands'),
        ('Amps',      'Motor current'),
        ('Period (s)', 'Wave period  —  one dot per cycle  (IMU2 boat reference)'),
    ]
    for ax, (ylabel, title) in zip(axes, labels):
        ax.set_ylabel(ylabel, color='white', fontsize=8)
        ax.set_title(title, color='#aaaaaa', fontsize=9, pad=2)
        ax.legend(fontsize=7, loc='upper right', facecolor='#1a1a2e', labelcolor='white')
        ax.grid(True, alpha=0.2, color='#444')
        ax.axhline(0, color='#555', lw=0.5)

    axes[4].set_xlabel('Time (s)', color='white', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def _update(_frame):
        t  = list(_buf['t'])
        if len(t) < 20:
            return l_r2, l_r1, l_p2, l_p1, l_cr, l_cp, l_ra, l_pa, l_pr, l_pp

        r1 = np.array(_buf['r1']); p1 = np.array(_buf['p1'])
        r2 = np.array(_buf['r2']); p2 = np.array(_buf['p2'])
        cr = np.array(_buf['cr']); cp = np.array(_buf['cp'])
        ra = np.array(_buf['ra']); pa = np.array(_buf['pa'])
        t_arr = np.array(t)

        l_r2.set_data(t, r2); l_r1.set_data(t, r1)
        l_p2.set_data(t, p2); l_p1.set_data(t, p1)
        l_cr.set_data(t, cr); l_cp.set_data(t, cp)
        l_ra.set_data(t, ra); l_pa.set_data(t, pa)

        t0, t1 = t[0], t[-1]
        dt_med = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else LOOP_PERIOD

        # ── Per-cycle wave period via upward zero-crossing on IMU2 ───────
        MIN_PERIOD = 0.3   # ignore cycles faster than 0.3s (noise)
        MAX_PERIOD = 15.0  # ignore cycles slower than 15s

        r2_dc = r2 - r2.mean();  p2_dc = p2 - p2.mean()
        # Find where new samples start (after last processed time), overlap by 2 samples
        last_t = _zc_state['last_t']
        new_idx = np.searchsorted(t_arr, last_t)
        scan_start = max(1, new_idx - 2)
        _zc_state['last_t'] = t_arr[-1]

        for k in range(scan_start, len(t_arr) - 1):
            tk, tk1 = t_arr[k], t_arr[k+1]
            # Roll upward zero-crossing
            if r2_dc[k] <= 0 and r2_dc[k+1] > 0:
                frac = -r2_dc[k] / (r2_dc[k+1] - r2_dc[k])
                tc = tk + frac * (tk1 - tk)
                amp = np.abs(r2_dc[max(0, k-10):k+10]).max()
                if amp > 0.3 and _zc_state['r_t_last'] is not None:
                    T = tc - _zc_state['r_t_last']
                    if MIN_PERIOD < T < MAX_PERIOD:
                        _period_buf_t.append(tc)
                        _period_buf_roll.append(T)
                        _period_buf_ptch.append(None)
                _zc_state['r_t_last'] = tc
            # Pitch upward zero-crossing
            if p2_dc[k] <= 0 and p2_dc[k+1] > 0:
                frac = -p2_dc[k] / (p2_dc[k+1] - p2_dc[k])
                tc = tk + frac * (tk1 - tk)
                amp = np.abs(p2_dc[max(0, k-10):k+10]).max()
                if amp > 0.3 and _zc_state['p_t_last'] is not None:
                    T = tc - _zc_state['p_t_last']
                    if MIN_PERIOD < T < MAX_PERIOD:
                        _period_buf_t.append(tc)
                        _period_buf_roll.append(None)
                        _period_buf_ptch.append(T)
                _zc_state['p_t_last'] = tc

        # Build separate t/val lists for roll and pitch (skip Nones)
        pt_all = list(_period_buf_t)
        pr_all = list(_period_buf_roll)
        pp_all = list(_period_buf_ptch)
        pt_r = [pt_all[i] for i in range(len(pt_all)) if pr_all[i] is not None]
        pv_r = [pr_all[i] for i in range(len(pr_all)) if pr_all[i] is not None]
        pt_p = [pt_all[i] for i in range(len(pt_all)) if pp_all[i] is not None]
        pv_p = [pp_all[i] for i in range(len(pp_all)) if pp_all[i] is not None]

        if pt_r: l_pr.set_data(pt_r, pv_r)
        if pt_p: l_pp.set_data(pt_p, pv_p)

        if pt_r or pt_p:
            axes[4].set_xlim(t0, t1)
            all_vals = pv_r + pv_p
            if all_vals:
                axes[4].set_ylim(0, min(MAX_PERIOD, max(all_vals) * 1.3 + 0.5))
            last_r = f"{pv_r[-1]:.2f}s" if pv_r else "—"
            last_p = f"{pv_p[-1]:.2f}s" if pv_p else "—"
            axes[4].set_title(
                f"Wave period (IMU2 boat reference)  —  "
                f"Roll: {last_r}   Pitch: {last_p}",
                color='#aaaaaa', fontsize=9, pad=2)

        # Auto-scale y
        def _yscale(ax, *arrs):
            vals = np.concatenate([np.array(a) for a in arrs])
            if len(vals) == 0: return
            lo, hi = vals.min(), vals.max()
            pad = max((hi - lo) * 0.15, 0.5)
            ax.set_ylim(lo - pad, hi + pad)

        for ax in axes[:4]:
            ax.set_xlim(t0, t1)
        _yscale(axes[0], r2, r1)
        _yscale(axes[1], p2, p1)
        _yscale(axes[2], cr, cp)
        _yscale(axes[3], ra, pa, np.array([0, 4.5]))

        # Live stats in titles
        rr2 = np.sqrt(np.mean(r2**2)) or 1e-9
        rr1 = np.sqrt(np.mean(r1**2))
        rp2 = np.sqrt(np.mean(p2**2)) or 1e-9
        rp1 = np.sqrt(np.mean(p1**2))
        axes[0].set_title(
            f"Roll  —  {(1-rr1/rr2)*100:.0f}% reduction  |  "
            f"Kp={gains.get('ROLL_GAIN','?')} Kd={gains.get('ROLL_KD','?')} Kff={gains.get('ROLL_KFF','?')}",
            color='#aaaaaa', fontsize=9, pad=2)
        axes[1].set_title(
            f"Pitch —  {(1-rp1/rp2)*100:.0f}% reduction  |  "
            f"Kp={gains.get('PITCH_GAIN','?')} Kd={gains.get('PITCH_KD','?')} Kff={gains.get('PITCH_KFF','?')}",
            color='#aaaaaa', fontsize=9, pad=2)

        if _ctrl_fault.is_set() or _ctrl_stop.is_set():
            fig.suptitle("STOPPED — control loop exited", color='red',
                         fontsize=12, fontweight='bold')

        return l_r2, l_r1, l_p2, l_p1, l_cr, l_cp, l_ra, l_pa, l_pr, l_pp

    ani = animation.FuncAnimation(fig, _update, interval=100, blit=False, cache_frame_data=False)

    plt.show()   # blocks until window closed or Ctrl+C via signal handler
    _ctrl_stop.set()
    ctrl_thread.join(timeout=3.0)
