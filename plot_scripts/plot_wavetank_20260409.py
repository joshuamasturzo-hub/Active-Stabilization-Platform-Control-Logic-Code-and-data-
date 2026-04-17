"""Combined comparison plot for the 4 wavetank tests — 2026-04-09."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

FILES = [
    "/home/edg5/captone_OG/fpd_tune_data_20260409_173209.csv",
    "/home/edg5/captone_OG/fpd_tune_data_20260409_173617.csv",
    "/home/edg5/captone_OG/fpd_tune_data_20260409_173850.csv",
    "/home/edg5/captone_OG/fpd_tune_data_20260409_174117.csv",
]

LABELS = ["Test 1\n17:32", "Test 2\n17:36", "Test 3\n17:38", "Test 4\n17:41"]
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

datasets = []
for f in FILES:
    df = pd.read_csv(f)
    datasets.append(df)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#f8f9fa")

# Row 0: Roll angle (all 4 tests)
# Row 1: Pitch angle (all 4 tests)
# Row 2: Motor commands roll+pitch (all 4 tests)
# Row 3: Summary bar chart (RMS reduction)
outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                          height_ratios=[2, 2, 2, 1.5])

# ── Rows 0-2: time-series panels ──────────────────────────────────────────────
row_specs = [
    ("Roll Angle (deg)",  "imu2_roll_deg",  "imu1_roll_deg",  "cmd_roll_erpm"),
    ("Pitch Angle (deg)", "imu2_pitch_deg", "imu1_pitch_deg", "cmd_pitch_erpm"),
]

for row_idx, (ylabel, ref_col, plat_col, cmd_col) in enumerate(row_specs):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_idx],
                                            wspace=0.28)
    for col_idx, (df, label, color) in enumerate(zip(datasets, LABELS, COLORS)):
        ax = fig.add_subplot(inner[col_idx])
        t  = df["t_s"].values
        ref  = df[ref_col].values
        plat = df[plat_col].values

        ax.plot(t, ref,  color="steelblue", lw=0.8, alpha=0.65, label="IMU2 (boat)")
        ax.plot(t, plat, color="crimson",   lw=1.0, alpha=0.9,  label="IMU1 (platform)")
        ax.axhline(0, color="k", lw=0.5, ls="--")

        rms_ref  = np.sqrt(np.mean(ref**2))
        rms_plat = np.sqrt(np.mean(plat**2))
        red = (1 - rms_plat / rms_ref) * 100 if rms_ref > 0 else 0

        axis_name = "Roll" if "roll" in ref_col else "Pitch"
        ax.set_title(f"{label}\n{axis_name} RMS: {rms_plat:.2f}° ({red:.1f}% ↓)",
                     fontsize=9, color=color, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if col_idx == 0 and row_idx == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_xlim(0, t[-1])

# ── Row 2: Motor commands ─────────────────────────────────────────────────────
inner_cmd = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.28)
for col_idx, (df, label, color) in enumerate(zip(datasets, LABELS, COLORS)):
    ax = fig.add_subplot(inner_cmd[col_idx])
    t  = df["t_s"].values
    cr = df["cmd_roll_erpm"].values
    cp = df["cmd_pitch_erpm"].values
    ax.plot(t, cr, color="royalblue", lw=0.8, alpha=0.85, label="Roll cmd")
    ax.plot(t, cp, color="darkorange", lw=0.8, alpha=0.85, label="Pitch cmd")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"{label}\nMotor Commands", fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    if col_idx == 0:
        ax.set_ylabel("Command (ERPM)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_xlim(0, t[-1])

# ── Row 3: Summary bar chart ──────────────────────────────────────────────────
ax_sum = fig.add_subplot(outer[3])
test_names = ["Test 1\n(17:32)", "Test 2\n(17:36)", "Test 3\n(17:38)", "Test 4\n(17:41)"]
roll_reductions  = []
pitch_reductions = []
roll_rms_plat  = []
pitch_rms_plat = []

for df in datasets:
    r2 = np.sqrt(np.mean(df["imu2_roll_deg"].values**2))
    r1 = np.sqrt(np.mean(df["imu1_roll_deg"].values**2))
    p2 = np.sqrt(np.mean(df["imu2_pitch_deg"].values**2))
    p1 = np.sqrt(np.mean(df["imu1_pitch_deg"].values**2))
    roll_reductions.append((1 - r1/r2)*100 if r2 > 0 else 0)
    pitch_reductions.append((1 - p1/p2)*100 if p2 > 0 else 0)
    roll_rms_plat.append(r1)
    pitch_rms_plat.append(p1)

x = np.arange(4)
w = 0.35
bars_r = ax_sum.bar(x - w/2, roll_reductions,  w, label="Roll reduction %",
                    color="royalblue", alpha=0.85, edgecolor="navy")
bars_p = ax_sum.bar(x + w/2, pitch_reductions, w, label="Pitch reduction %",
                    color="darkorange", alpha=0.85, edgecolor="saddlebrown")

for bar, rms in zip(bars_r, roll_rms_plat):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{rms:.2f}°\nRMS", ha="center", va="bottom", fontsize=7.5, color="navy")
for bar, rms in zip(bars_p, pitch_rms_plat):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{rms:.2f}°\nRMS", ha="center", va="bottom", fontsize=7.5, color="saddlebrown")

ax_sum.set_xticks(x)
ax_sum.set_xticklabels(test_names, fontsize=9)
ax_sum.set_ylabel("Vibration Reduction (%)", fontsize=10)
ax_sum.set_title("Performance Summary — All 4 Wavetank Tests (2026-04-09)", fontsize=11, fontweight="bold")
ax_sum.legend(fontsize=9)
ax_sum.set_ylim(0, 110)
ax_sum.axhline(90, color="green", ls="--", lw=0.8, alpha=0.6, label="90% target")
ax_sum.grid(True, axis="y", alpha=0.3)

fig.suptitle("Wavetank Tests — FPD Head Tune — 2026-04-09\n"
             "Blue=boat (IMU2), Red=platform (IMU1)",
             fontsize=13, fontweight="bold", y=0.995)

out = "/home/edg5/captone_OG/wavetank_20260409_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
