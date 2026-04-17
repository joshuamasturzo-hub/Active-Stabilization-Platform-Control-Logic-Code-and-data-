import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

TESTS = [
    ("pd_control_data_20260403_183150.csv", "Earlier PD test (18:32)"),
    ("pd_control_data_20260403_200031.csv", "Best PD test (20:00)"),
]
OUT = "pd_comparison — test12 vs earlier.png"

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("PD Control Comparison — Roll P=600 D=15 | Pitch P=300 D=8 | EMA filtered",
             fontsize=12, fontweight="bold")

colors = [("royalblue", "tomato"), ("seagreen", "darkorange")]

for col, (csv, label) in enumerate(TESTS):
    df = pd.read_csv(csv)
    t  = df["t_s"].values
    r1 = df["imu1_roll_deg"].values; p1 = df["imu1_pitch_deg"].values
    r2 = df["imu2_roll_deg"].values; p2 = df["imu2_pitch_deg"].values
    rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
    rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
    red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
    c_ref, c_plt = colors[col]

    ax = axes[0][col]
    ax.plot(t, r2, color=c_ref, alpha=0.6, lw=1.0, label="IMU2 Ref")
    ax.plot(t, r1, color=c_plt, alpha=0.9, lw=1.2, label="IMU1 Platform")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.fill_between(t, -0.5, 0.5, alpha=0.10, color="green")
    ax.set_title(f"{label}\nRoll: IMU2={rms_r2:.2f}° IMU1={rms_r1:.2f}° Reduction={red_r:.1f}%", fontsize=9)
    ax.set_ylabel("Roll (deg)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1][col]
    ax.plot(t, p2, color=c_ref, alpha=0.6, lw=1.0, label="IMU2 Ref")
    ax.plot(t, p1, color=c_plt, alpha=0.9, lw=1.2, label="IMU1 Platform")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.fill_between(t, -1.0, 1.0, alpha=0.10, color="green")
    ax.set_title(f"Pitch: IMU2={rms_p2:.2f}° IMU1={rms_p1:.2f}° Reduction={red_p:.1f}%", fontsize=9)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for row in range(2):
        axes[row][col].set_xlim(t[0], t[-1])

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
