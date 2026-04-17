import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260406_223747.csv"
OUT = "fpd_analysis_20260406_223747.png"

df = pd.read_csv(CSV)
t  = df["t_s"].values
r1 = df["imu1_roll_deg"].values;  p1 = df["imu1_pitch_deg"].values
r2 = df["imu2_roll_deg"].values;  p2 = df["imu2_pitch_deg"].values
cr = df["cmd_roll_erpm"].values;  cp = df["cmd_pitch_erpm"].values
dr = df["d_roll_erpm"].values;    dp = df["d_pitch_erpm"].values
fr = df["ff_roll_erpm"].values;   fp = df["ff_pitch_erpm"].values
ra = df["motor_roll_amps"].values; pa = df["motor_pitch_amps"].values

n = len(t); dt = np.median(np.diff(t))
freqs = rfftfreq(n, d=dt)
def dom_f(s):
    m = np.abs(rfft(s - s.mean())); return freqs[np.argmax(m[1:])+1]

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
p1_f = dom_f(p1); p2_f = dom_f(p2)

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle("FPD Analysis — 20260406_223747 | PITCH_GAIN=500 (too high) PITCH_KD=-20",
             fontsize=12, fontweight="bold", y=0.99)

ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, r1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -0.5, 0.5, alpha=0.10, color="green", label="±0.5° deadband")
ax.set_ylabel("Roll (deg)"); ax.grid(True, alpha=0.3)
ax.set_title(f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%  "
             f"Dominant={dom_f(r2):.3f} Hz  Roll amps max={ra.max():.2f}A", fontsize=9)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, p1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform (oscillating)")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -1.0, 1.0, alpha=0.10, color="green", label="±1.0° (recommended)")
ax.set_ylabel("Pitch (deg)"); ax.grid(True, alpha=0.3)
ax.set_title(f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%  "
             f"IMU2 dominant={p2_f:.3f} Hz BUT IMU1 oscillates at {p1_f:.3f} Hz — self-induced", fontsize=9)
ax.legend(fontsize=8, loc="upper right")

ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.8, lw=1.0, label="Roll cmd"); ax.plot(t, cp, "m-", alpha=0.8, lw=1.0, label="Pitch cmd")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax2.plot(t, ra, "g--", alpha=0.6, lw=0.9, label="Roll amps")
ax2.plot(t, pa, "m--", alpha=0.6, lw=0.9, label="Pitch amps")
ax2.axhline(4.5, color="red", lw=1.0, ls=":", label="4.5A rated")
ax.set_ylabel("ERPM"); ax2.set_ylabel("Current (A)")
ax.set_title(f"Commands | Roll max={np.abs(cr).max():.0f} ERPM | Pitch max={np.abs(cp).max():.0f} ERPM | "
             f"Roll amps {ra.max():.2f}A  Pitch {pa.max():.2f}A", fontsize=9)
lines1,l1=ax.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(t, dr, "g-", lw=1.0, alpha=0.8, label=f"D roll (max={np.abs(dr).max():.0f})")
ax.plot(t, dp, "m-", lw=1.0, alpha=0.8, label=f"D pitch (max={np.abs(dp).max():.0f})")
ax.plot(t, fr, "g--", lw=0.9, alpha=0.6, label=f"FF roll (max={np.abs(fr).max():.0f})")
ax.plot(t, fp, "m--", lw=0.9, alpha=0.6, label=f"FF pitch (max={np.abs(fp).max():.0f})")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
ax.set_title("D and FF term contributions", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for a in axes: a.set_xlim(t[0], t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
