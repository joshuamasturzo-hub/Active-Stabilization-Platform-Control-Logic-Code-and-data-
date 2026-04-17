import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "ff_control_data_20260403_183937.csv"
OUT = "ff test 1 — Roll Kp600 Kff8 Pitch Kp300 Kff5 100Hz.png"

df = pd.read_csv(CSV)
t  = df["t_s"].values
r1 = df["imu1_roll_deg"].values
p1 = df["imu1_pitch_deg"].values
r2 = df["imu2_roll_deg"].values
p2 = df["imu2_pitch_deg"].values
cr = df["cmd_roll_erpm"].values
cp = df["cmd_pitch_erpm"].values
fr = df["ff_roll_erpm"].values
fp = df["ff_pitch_erpm"].values
ra = df["motor_roll_amps"].values
pa = df["motor_pitch_amps"].values
rr = df["imu2_roll_rate_dps"].values
pr = df["imu2_pitch_rate_dps"].values

n=len(t); dt=np.median(np.diff(t))
freqs=rfftfreq(n,d=dt)
def dominant_freq(s): return freqs[np.argmax(np.abs(rfft(s-s.mean()))[1:])+1]
def lag_ms(ref,sig):
    corr=np.correlate(ref-ref.mean(),sig-sig.mean(),mode='full')
    lags=np.arange(-(n-1),n)*dt
    return lags[np.argmax(corr)]*1000

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
r2_f=dominant_freq(r2); roll_lag=lag_ms(r2,r1)

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle("FF Test 1 — Feedforward+P | Roll Kp=600 Kff=8 | Pitch Kp=300 Kff=5 | 100 Hz",
             fontsize=12, fontweight="bold", y=0.99)

ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference (boat)")
ax.plot(t, r1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -0.5, 0.5, alpha=0.10, color="green", label="±0.5° deadband")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%  "
             f"Freq={r2_f:.3f} Hz  Lag={roll_lag:.0f}ms | Best P-only: 85.8%", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference (boat)")
ax.plot(t, p1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -1.0, 1.0, alpha=0.10, color="green", label="±1.0° deadband")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%  "
             f"Last 4s stable | Best P-only: 67.9%", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.8, lw=1.0, label="Roll cmd ERPM")
ax.plot(t, cp, "m-", alpha=0.8, lw=1.0, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax2.plot(t, ra, "g--", alpha=0.6, lw=0.9, label="Roll amps")
ax2.plot(t, pa, "m--", alpha=0.6, lw=0.9, label="Pitch amps")
ax2.axhline(4.0, color="red", lw=0.8, ls=":", label="4.0A limit")
ax2.axhline(4.63, color="red", lw=1.2, ls="-", alpha=0.5, label=f"Roll peak 4.63A !")
ax.set_ylabel("ERPM", fontsize=10); ax2.set_ylabel("Current (A)", fontsize=10)
ax.set_title(f"Commands & Current | Roll max {np.abs(cr).max():.0f} ERPM {ra.max():.2f}A ⚠ | "
             f"Pitch max {np.abs(cp).max():.0f} ERPM {pa.max():.2f}A", fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[3]
ax2 = ax.twinx()
ax.plot(t, fr, "g-", alpha=0.85, lw=1.2, label=f"FF roll (max={np.abs(fr).max():.0f} ERPM mean={np.abs(fr).mean():.0f})")
ax.plot(t, fp, "m-", alpha=0.85, lw=1.2, label=f"FF pitch (max={np.abs(fp).max():.0f} ERPM mean={np.abs(fp).mean():.0f})")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax2.plot(t, rr, "b-", alpha=0.4, lw=0.8, label="IMU2 roll rate (dps)")
ax2.plot(t, pr, "r-", alpha=0.4, lw=0.8, label="IMU2 pitch rate (dps)")
ax.set_ylabel("FF contribution (ERPM)", fontsize=10)
ax2.set_ylabel("IMU2 rate (°/s)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title("Feedforward Term — KFF too small, barely contributing vs P term. Increase KFF next run.", fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for ax in axes: ax.set_xlim(t[0], t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
