import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "vel_control_data_20260403_193437.csv"
OUT = "dry test 9 data plot — PD Roll800kd20 Pitch300kd5.png"

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

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100

n = len(t)
t_osc = t[3*n//4]

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle("Dry Test 9 — PD | Roll P=800 D=20 | Pitch P=300 D=5 | D cap 1500",
             fontsize=12, fontweight="bold", y=0.99)

ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, r1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -0.5, 0.5, alpha=0.10, color="green", label="±0.5° deadband")
ax.axvspan(t_osc, t[-1], alpha=0.10, color="red", label="Oscillation zone")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.6, lw=1.0, label="IMU2 Reference")
ax.plot(t, p1, "r-", alpha=0.9, lw=1.2, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, -1.0, 1.0, alpha=0.10, color="green", label="±1.0° deadband")
ax.axvspan(t_osc, t[-1], alpha=0.10, color="red", label="Oscillation zone")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.8, lw=1.0, label="Roll cmd ERPM")
ax.plot(t, cp, "m-", alpha=0.8, lw=1.0, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvspan(t_osc, t[-1], alpha=0.10, color="red")
ax2.plot(t, ra, "g--", alpha=0.6, lw=0.9, label="Roll amps")
ax2.plot(t, pa, "m--", alpha=0.6, lw=0.9, label="Pitch amps")
ax2.axhline(4.0, color="red", lw=0.8, ls=":", label="4.0A limit")
ax.set_ylabel("ERPM", fontsize=10); ax2.set_ylabel("Current (A)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title(f"Motor Commands & Current | Roll max {np.abs(cr).max():.0f} ERPM {ra.max():.2f}A | Pitch max {np.abs(cp).max():.0f} ERPM {pa.max():.2f}A", fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(t[0], t[-1])

plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
