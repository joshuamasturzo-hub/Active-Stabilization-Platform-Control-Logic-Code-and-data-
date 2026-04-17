"""Regenerate dry test 5 plots with instantaneous period via Hilbert transform."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq

CSV = "vel_control_data.csv"
OUT_MAIN = "dry test 5 data plot — high motion Roll500 Pitch400 ERPM5000.png"
OUT_FREQ = "dry test 5 frequency analysis.png"

df = pd.read_csv(CSV)
t = df["t_s"].values
imu1_roll  = df["imu1_roll_deg"].values
imu1_pitch = df["imu1_pitch_deg"].values
imu2_roll  = df["imu2_roll_deg"].values
imu2_pitch = df["imu2_pitch_deg"].values
cmd_roll   = df["cmd_roll_erpm"].values
cmd_pitch  = df["cmd_pitch_erpm"].values
mot_roll_pos  = df["motor_roll_pos_deg"].values
mot_pitch_pos = df["motor_pitch_pos_deg"].values

dt = np.median(np.diff(t))
fs = 1.0 / dt

def lowpass(sig, cutoff=3.0, fs=fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)

# Detrend + filter before Hilbert (removes DC offset and slow drift)
imu2_roll_centered  = lowpass(detrend(imu2_roll))
imu2_pitch_centered = lowpass(detrend(imu2_pitch))

def inst_period(sig, fs, smooth_cutoff=0.8):
    analytic = hilbert(sig)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.diff(phase) / (2.0 * np.pi / fs)
    inst_f = np.append(inst_f, inst_f[-1])
    inst_f = np.clip(np.abs(inst_f), 0.05, 5.0)
    inst_p = 1.0 / inst_f
    # light smoothing
    inst_p = lowpass(inst_p, cutoff=smooth_cutoff)
    return inst_p

roll_inst_p  = inst_period(imu2_roll_centered,  fs)
pitch_inst_p = inst_period(imu2_pitch_centered, fs)

# FFT for dominant frequency
n = len(imu2_roll)
freqs = rfftfreq(n, d=dt)
roll_mag  = np.abs(rfft(detrend(imu2_roll)))
pitch_mag = np.abs(rfft(detrend(imu2_pitch)))

roll_peak_f  = freqs[np.argmax(roll_mag[1:]) + 1]
pitch_peak_f = freqs[np.argmax(pitch_mag[1:]) + 1]
roll_peak_p  = 1.0 / roll_peak_f  if roll_peak_f > 0 else 0
pitch_peak_p = 1.0 / pitch_peak_f if pitch_peak_f > 0 else 0

roll_amp  = np.std(imu2_roll)
pitch_amp = np.std(imu2_pitch)

def sea_state(amp):
    if amp < 0.5:  return "Calm (< SS1)"
    if amp < 1.5:  return "SS1 — Light"
    if amp < 3.0:  return "SS2 — Slight"
    if amp < 6.0:  return "SS3 — Moderate"
    return           "SS4+ — Rough"

roll_ss  = sea_state(roll_amp)
pitch_ss = sea_state(pitch_amp)

roll_rms_imu1  = np.sqrt(np.mean(imu1_roll**2))
roll_rms_imu2  = np.sqrt(np.mean(imu2_roll**2))
pitch_rms_imu1 = np.sqrt(np.mean(imu1_pitch**2))
pitch_rms_imu2 = np.sqrt(np.mean(imu2_pitch**2))
roll_reduction  = (1 - roll_rms_imu1  / roll_rms_imu2)  * 100 if roll_rms_imu2  > 0 else 0
pitch_reduction = (1 - pitch_rms_imu1 / pitch_rms_imu2) * 100 if pitch_rms_imu2 > 0 else 0

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4-panel stabilisation plot
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle(
    "Dry Test 5 — High Motion | Roll gain 500 | Pitch gain 400 | MAX_ERPM 5000",
    fontsize=13, fontweight="bold", y=0.98,
)

ax = axes[0]
ax.plot(t, imu2_roll,  "b-",  alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, imu1_roll,  "r-",  alpha=0.8, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(
    f"Roll | IMU2 RMS={roll_rms_imu2:.2f}° | IMU1 RMS={roll_rms_imu1:.2f}° | "
    f"Reduction={roll_reduction:.1f}% | Dominant freq={roll_peak_f:.3f} Hz | "
    f"Period={roll_peak_p:.2f}s | {roll_ss}",
    fontsize=8,
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, imu2_pitch, "b-",  alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, imu1_pitch, "r-",  alpha=0.8, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(
    f"Pitch | IMU2 RMS={pitch_rms_imu2:.2f}° | IMU1 RMS={pitch_rms_imu1:.2f}° | "
    f"Reduction={pitch_reduction:.1f}% | Dominant freq={pitch_peak_f:.3f} Hz | "
    f"Period={pitch_peak_p:.2f}s | {pitch_ss}",
    fontsize=8,
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t, cmd_roll,  "g-",  alpha=0.8, lw=1.0, label="Roll cmd ERPM")
ax.plot(t, cmd_pitch, "m-",  alpha=0.8, lw=1.0, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("ERPM", fontsize=10)
ax.set_title("Motor Commands", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(t, roll_inst_p,  "b-", alpha=0.85, lw=1.2, label="Roll instantaneous period")
ax.plot(t, pitch_inst_p, "r-", alpha=0.85, lw=1.2, label="Pitch instantaneous period")
ax.axhline(roll_peak_p,  color="b", lw=0.8, ls="--", alpha=0.5,
           label=f"Roll dominant {roll_peak_p:.2f}s")
ax.axhline(pitch_peak_p, color="r", lw=0.8, ls="--", alpha=0.5,
           label=f"Pitch dominant {pitch_peak_p:.2f}s")
ax.set_ylabel("Period (s)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title("Instantaneous Wave Period (Hilbert transform on detrended signal)", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 8)

for ax in axes:
    ax.set_xlim(t[0], t[-1])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_MAIN, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_MAIN}")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Frequency analysis
# ════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))
fig2.suptitle("Dry Test 5 — Frequency & Period Analysis (IMU2 Reference)", fontsize=12, fontweight="bold")

ax = axes2[0]
ax.plot(freqs[1:], roll_mag[1:],  "b-", lw=1.2, label="Roll spectrum")
ax.plot(freqs[1:], pitch_mag[1:], "r-", lw=1.2, label="Pitch spectrum")
ax.axvline(roll_peak_f,  color="b", ls="--", lw=1.0, label=f"Roll peak {roll_peak_f:.3f} Hz")
ax.axvline(pitch_peak_f, color="r", ls="--", lw=1.0, label=f"Pitch peak {pitch_peak_f:.3f} Hz")
ax.set_xlim(0, 3)
ax.set_xlabel("Frequency (Hz)", fontsize=10)
ax.set_ylabel("Magnitude", fontsize=10)
ax.set_title("FFT Spectrum of IMU2 (Reference Platform Motion)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(t, roll_inst_p,  "b-", lw=1.2, alpha=0.85, label="Roll instantaneous period")
ax.plot(t, pitch_inst_p, "r-", lw=1.2, alpha=0.85, label="Pitch instantaneous period")
ax.axhline(roll_peak_p,  color="b", ls="--", lw=0.8, alpha=0.6,
           label=f"Roll dominant {roll_peak_p:.2f}s")
ax.axhline(pitch_peak_p, color="r", ls="--", lw=0.8, alpha=0.6,
           label=f"Pitch dominant {pitch_peak_p:.2f}s")
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Period (s)", fontsize=10)
ax.set_title("Instantaneous Wave Period (Hilbert on detrended signal)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 8)
ax.set_xlim(t[0], t[-1])

ax = axes2[2]
ax.axis("off")
table_data = [
    ["Metric", "Roll Axis", "Pitch Axis"],
    ["Dominant frequency (FFT)", f"{roll_peak_f:.3f} Hz", f"{pitch_peak_f:.3f} Hz"],
    ["Dominant wave period", f"{roll_peak_p:.2f} s", f"{pitch_peak_p:.2f} s"],
    ["Amplitude (std dev)", f"{roll_amp:.2f}°", f"{pitch_amp:.2f}°"],
    ["Sea state estimate", roll_ss, pitch_ss],
    ["IMU2 RMS (reference)", f"{roll_rms_imu2:.2f}°", f"{pitch_rms_imu2:.2f}°"],
    ["IMU1 RMS (stabilized)", f"{roll_rms_imu1:.2f}°", f"{pitch_rms_imu1:.2f}°"],
    ["Reduction", f"{roll_reduction:.1f}%", f"{pitch_reduction:.1f}%"],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center", bbox=[0.1, 0.05, 0.8, 0.9])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2196F3")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")

plt.tight_layout()
plt.savefig(OUT_FREQ, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FREQ}")
