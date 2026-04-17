import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "vel_control_data_20260403_195311.csv"
OUT = "dry test 11 data plot — P-only Roll750 Pitch400 UNSTABLE.png"

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

n  = len(t)
dt = np.median(np.diff(t))
freqs = rfftfreq(n, d=dt)

def dominant_freq(sig):
    mag = np.abs(rfft(sig - sig.mean()))
    return freqs[np.argmax(mag[1:]) + 1]

r2_f = dominant_freq(r2); r1_f = dominant_freq(r1)
p2_f = dominant_freq(p2); p1_f = dominant_freq(p1)

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100

# find where oscillation starts — when IMU1 std exceeds IMU2 std in rolling window
window = 50
roll_ratio = np.array([r1[max(0,i-window):i].std() / (r2[max(0,i-window):i].std()+0.01)
                        for i in range(window, n)])
osc_start_idx = next((i for i,v in enumerate(roll_ratio) if v > 1.5), n) + window
t_osc = t[min(osc_start_idx, n-1)]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Dry Test 11 — P-only | Roll=750 | Pitch=400 | UNSTABLE — gains too high",
             fontsize=12, fontweight="bold", y=0.99)

ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, r1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvspan(t_osc, t[-1], alpha=0.12, color="red", label=f"Self-oscillation onset")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(
    f"Roll | Reduction={red_r:.1f}% | "
    f"Disturbance {r2_f:.3f} Hz → Platform oscillates at {r1_f:.3f} Hz (2× input!) | "
    f"Gain 750 exceeds stability limit", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, p1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvspan(t_osc, t[-1], alpha=0.12, color="red", label="Self-oscillation zone")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(
    f"Pitch | Reduction={red_p:.1f}% | "
    f"Platform oscillates at {p1_f:.3f} Hz with no input in last 3s | "
    f"Gain 400 exceeds stability limit", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.8, lw=1.0, label="Roll cmd ERPM")
ax.plot(t, cp, "m-", alpha=0.8, lw=1.0, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axhline(5000,  color="k", lw=0.6, ls=":", alpha=0.5, label="±5000 ERPM cap")
ax.axhline(-5000, color="k", lw=0.6, ls=":", alpha=0.5)
ax.axvspan(t_osc, t[-1], alpha=0.12, color="red")
ax2.plot(t, ra, "g--", alpha=0.6, lw=0.9, label="Roll amps")
ax2.plot(t, pa, "m--", alpha=0.6, lw=0.9, label="Pitch amps")
ax2.axhline(4.0, color="red", lw=0.8, ls=":", label="4.0A limit")
ax.set_ylabel("ERPM", fontsize=10); ax2.set_ylabel("Current (A)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title(f"Motor Commands & Current | Roll hit 5000 cap 10× | Amps: roll {ra.max():.2f}A pitch {pa.max():.2f}A", fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for ax in axes: ax.set_xlim(t[0], t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
