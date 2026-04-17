import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260408_151416.csv"
OUT = "pool test 1 — FPD Roll81pct Pitch57pct.png"

df = pd.read_csv(CSV)
t  = df["t_s"].values
r1 = df["imu1_roll_deg"].values
p1 = df["imu1_pitch_deg"].values
r2 = df["imu2_roll_deg"].values
p2 = df["imu2_pitch_deg"].values
cr = df["cmd_roll_erpm"].values
cp = df["cmd_pitch_erpm"].values
ra = df["motor_roll_amps"].values
pa = df["motor_pitch_amps"].values
ff_r = df["ff_roll_erpm"].values
ff_p = df["ff_pitch_erpm"].values

n = len(t); dt = np.median(np.diff(t))
freqs = rfftfreq(n, d=dt)

def dominant_freq(sig):
    mag = np.abs(rfft(sig - sig.mean()))
    return freqs[np.argmax(mag[1:])+1]

def zero_crossing_periods(sig, t):
    """Return (times, periods) at each upward zero crossing."""
    mean = sig.mean()
    s = sig - mean
    crossings = []
    for i in range(1, len(s)):
        if s[i-1] < 0 and s[i] >= 0:
            frac = -s[i-1] / (s[i] - s[i-1])
            crossings.append(t[i-1] + frac * (t[i] - t[i-1]))
    if len(crossings) < 2:
        return np.array([]), np.array([])
    times   = np.array([(crossings[i]+crossings[i+1])/2 for i in range(len(crossings)-1)])
    periods = np.diff(crossings)
    return times, periods

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
r2_f = dominant_freq(r2); p2_f = dominant_freq(p2)

roll_t,  roll_T  = zero_crossing_periods(r2, t)
pitch_t, pitch_T = zero_crossing_periods(p2, t)

fig, axes = plt.subplots(4, 1, figsize=(16, 15))
fig.suptitle("Pool Test 1 — FPD Control | Roll 81.2% reduction | Pitch 57.2% reduction",
             fontsize=13, fontweight="bold", y=0.99)

ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.55, lw=1.0, label="IMU2 Reference (boat)")
ax.plot(t, r1, "r-", alpha=0.90, lw=1.2, label="IMU1 Platform (stabilized)")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%  "
             f"Dominant freq={r2_f:.3f} Hz  Period={1/r2_f:.2f}s  "
             f"Peak IMU2={np.abs(r2).max():.1f}°  Peak IMU1={np.abs(r1).max():.1f}°", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.55, lw=1.0, label="IMU2 Reference (boat)")
ax.plot(t, p1, "r-", alpha=0.90, lw=1.2, label="IMU1 Platform (stabilized)")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%  "
             f"Dominant freq={p2_f:.3f} Hz  Period={1/p2_f:.2f}s  "
             f"Peak IMU2={np.abs(p2).max():.1f}°  Peak IMU1={np.abs(p1).max():.1f}°", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.7, lw=1.0, label="Roll cmd ERPM")
ax.plot(t, cp, "m-", alpha=0.7, lw=1.0, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax2.plot(t, ra, "g--", alpha=0.6, lw=0.8, label="Roll amps")
ax2.plot(t, pa, "m--", alpha=0.6, lw=0.8, label="Pitch amps")
ax2.axhline(4.5, color="red", lw=0.8, ls=":", label="4.5A rated")
ax.set_ylabel("ERPM", fontsize=10); ax2.set_ylabel("Current (A)", fontsize=10)
ax.set_title(f"Motor Commands & Current | Roll max {np.abs(cr).max():.0f} ERPM {ra.max():.2f}A | "
             f"Pitch max {np.abs(cp).max():.0f} ERPM {pa.max():.2f}A", fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[3]
if len(roll_T) > 0:
    ax.scatter(roll_t,  roll_T,  c="b", s=40, zorder=5, label=f"Roll period  (mean={roll_T.mean():.2f}s)")
    ax.axhline(roll_T.mean(),  color="b", lw=0.8, ls="--", alpha=0.5)
if len(pitch_T) > 0:
    ax.scatter(pitch_t, pitch_T, c="r", s=40, zorder=5, label=f"Pitch period (mean={pitch_T.mean():.2f}s)")
    ax.axhline(pitch_T.mean(), color="r", lw=0.8, ls="--", alpha=0.5)
ax.set_ylabel("Wave Period (s)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title("Wave Period — zero-crossing method (one point per cycle)", fontsize=9)
ax.set_ylim(0, max(
    roll_T.max() if len(roll_T) > 0 else 5,
    pitch_T.max() if len(pitch_T) > 0 else 5
) * 1.2)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for ax in axes: ax.set_xlim(t[0], t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
