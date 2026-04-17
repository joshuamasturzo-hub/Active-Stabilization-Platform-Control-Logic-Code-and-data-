import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260408_154954.csv"
OUT = "pool test 2 — FPD Roll82pct Pitch66pct 153s.png"

df  = pd.read_csv(CSV)
t   = df["t_s"].values
r1  = df["imu1_roll_deg"].values
p1  = df["imu1_pitch_deg"].values
r2  = df["imu2_roll_deg"].values
p2  = df["imu2_pitch_deg"].values
cr  = df["cmd_roll_erpm"].values
cp  = df["cmd_pitch_erpm"].values
ra  = df["motor_roll_amps"].values
pa  = df["motor_pitch_amps"].values
ff_r = df["ff_roll_erpm"].values
ff_p = df["ff_pitch_erpm"].values
d_r  = df["d_roll_erpm"].values
d_p  = df["d_pitch_erpm"].values

n  = len(t)
dt = np.median(np.diff(t))

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100

# ── Zero-crossing period detection ───────────────────────────────────────────
def zero_crossing_periods(sig, t, min_period=0.3, max_period=15.0):
    """Find upward zero crossings, return (mid_time, period) for each cycle."""
    sig_d = sig - sig.mean()
    crossings = []
    for i in range(1, len(sig_d)):
        if sig_d[i-1] < 0 and sig_d[i] >= 0:
            # linear interpolate exact crossing time
            frac = -sig_d[i-1] / (sig_d[i] - sig_d[i-1])
            crossings.append(t[i-1] + frac * (t[i] - t[i-1]))
    periods, mid_times = [], []
    for i in range(1, len(crossings)):
        p = crossings[i] - crossings[i-1]
        if min_period <= p <= max_period:
            periods.append(p)
            mid_times.append((crossings[i] + crossings[i-1]) / 2)
    return np.array(mid_times), np.array(periods)

roll_pt,  roll_per  = zero_crossing_periods(r2, t)
pitch_pt, pitch_per = zero_crossing_periods(p2, t)

fig, axes = plt.subplots(5, 1, figsize=(16, 18))
fig.suptitle(
    f"Pool Test 2 — FPD Control | 153s | Roll {red_r:.1f}% | Pitch {red_p:.1f}% | 200 Hz loop",
    fontsize=13, fontweight="bold", y=0.99,
)

# Panel 1 — Roll
ax = axes[0]
ax.plot(t, r2, "b-", alpha=0.5, lw=0.8, label="IMU2 Reference")
ax.plot(t, r1, "r-", alpha=0.85, lw=1.0, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Roll (deg)", fontsize=10)
ax.set_title(
    f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%  "
    f"Peak disturbance={np.abs(r2).max():.1f}°  Peak platform={np.abs(r1).max():.1f}°",
    fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# Panel 2 — Pitch
ax = axes[1]
ax.plot(t, p2, "b-", alpha=0.5, lw=0.8, label="IMU2 Reference")
ax.plot(t, p1, "r-", alpha=0.85, lw=1.0, label="IMU1 Platform")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Pitch (deg)", fontsize=10)
ax.set_title(
    f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%  "
    f"Peak disturbance={np.abs(p2).max():.1f}°  Peak platform={np.abs(p1).max():.1f}°",
    fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# Panel 3 — Commands + Amps
ax = axes[2]
ax2 = ax.twinx()
ax.plot(t, cr, "g-", alpha=0.7, lw=0.8, label="Roll cmd ERPM")
ax.plot(t, cp, "m-", alpha=0.7, lw=0.8, label="Pitch cmd ERPM")
ax.axhline(0, color="k", lw=0.4, ls="--")
ax.axhline(5000,  color="k", lw=0.5, ls=":", alpha=0.4)
ax.axhline(-5000, color="k", lw=0.5, ls=":", alpha=0.4, label="±5000 cap")
ax2.plot(t, ra, "g--", alpha=0.5, lw=0.7, label=f"Roll amps (max={ra.max():.1f}A)")
ax2.plot(t, pa, "m--", alpha=0.5, lw=0.7, label=f"Pitch amps (max={pa.max():.1f}A)")
ax2.axhline(4.0, color="red", lw=0.8, ls=":", label="4.0A cont. limit")
ax.set_ylabel("ERPM", fontsize=10); ax2.set_ylabel("Current (A)", fontsize=10)
ax.set_title(
    f"Commands & Current | FF roll max={np.abs(ff_r).max():.0f} ERPM  FF pitch max={np.abs(ff_p).max():.0f} ERPM  "
    f"D roll max={np.abs(d_r).max():.0f}  D pitch max={np.abs(d_p).max():.0f}",
    fontsize=9)
lines1,labs1=ax.get_legend_handles_labels()
lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# Panel 4 — Roll period scatter
ax = axes[3]
if len(roll_per) > 0:
    ax.scatter(roll_pt, roll_per, color="b", s=30, alpha=0.8, zorder=3, label=f"Roll period ({len(roll_per)} cycles)")
    ax.axhline(np.median(roll_per), color="b", lw=1.0, ls="--",
               label=f"Median {np.median(roll_per):.2f}s  ({1/np.median(roll_per):.3f} Hz)")
    ax.set_ylim(0, min(15, roll_per.max()*1.3))
ax.set_ylabel("Period (s)", fontsize=10)
ax.set_title(f"Roll Wave Period — {len(roll_per)} cycles detected | "
             f"Median={np.median(roll_per):.2f}s | Min={roll_per.min():.2f}s | Max={roll_per.max():.2f}s"
             if len(roll_per)>0 else "Roll Wave Period — no cycles detected", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# Panel 5 — Pitch period scatter
ax = axes[4]
if len(pitch_per) > 0:
    ax.scatter(pitch_pt, pitch_per, color="m", s=30, alpha=0.8, zorder=3, label=f"Pitch period ({len(pitch_per)} cycles)")
    ax.axhline(np.median(pitch_per), color="m", lw=1.0, ls="--",
               label=f"Median {np.median(pitch_per):.2f}s  ({1/np.median(pitch_per):.3f} Hz)")
    ax.set_ylim(0, min(15, pitch_per.max()*1.3))
ax.set_ylabel("Period (s)", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_title(f"Pitch Wave Period — {len(pitch_per)} cycles detected | "
             f"Median={np.median(pitch_per):.2f}s | Min={pitch_per.min():.2f}s | Max={pitch_per.max():.2f}s"
             if len(pitch_per)>0 else "Pitch Wave Period — no cycles detected", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(t[0], t[-1])

plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
