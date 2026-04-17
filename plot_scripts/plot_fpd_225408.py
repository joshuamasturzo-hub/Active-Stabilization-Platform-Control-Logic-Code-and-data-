import numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260406_225408.csv"
OUT = CSV.replace(".csv", "_analysis.png")

df = pd.read_csv(CSV)
t  = df["t_s"].values
r1 = df["imu1_roll_deg"].values;  p1 = df["imu1_pitch_deg"].values
r2 = df["imu2_roll_deg"].values;  p2 = df["imu2_pitch_deg"].values
cr = df["cmd_roll_erpm"].values;  cp = df["cmd_pitch_erpm"].values
ir = df["i_roll_erpm"].values;    ip = df["i_pitch_erpm"].values
dr = df["d_roll_erpm"].values;    dp = df["d_pitch_erpm"].values
fr = df["ff_roll_erpm"].values;   fp = df["ff_pitch_erpm"].values
ra = df["motor_roll_amps"].values; pa = df["motor_pitch_amps"].values
pr = cr - ir - dr - fr;           pp = cp - ip - dp - fp

n = len(t); dt_med = float(np.median(np.diff(t)))

# ── Compute median oscillation period within a signal segment ────────────
def median_period_seg(sig, t_arr, min_amp=0.3):
    """Upward zero-crossing detection → median period. Returns None if < 2 cycles."""
    s = sig - np.mean(sig)
    crossings = []
    for i in range(len(s) - 1):
        if s[i] <= 0 and s[i+1] > 0:
            frac = -s[i] / (s[i+1] - s[i])
            crossings.append(t_arr[i] + frac * (t_arr[i+1] - t_arr[i]))
    periods = []
    for k in range(len(crossings) - 1):
        t0c, t1c = crossings[k], crossings[k+1]
        period = t1c - t0c
        mask = (t_arr >= t0c) & (t_arr <= t1c)
        if mask.sum() < 2: continue
        amp = np.abs(s[mask]).max()
        if amp < min_amp: continue
        periods.append(period)
    if len(periods) < 1:
        return None
    return float(np.median(periods))

# ── Quarter stats ─────────────────────────────────────────────────────────
qs   = [0, n//4, n//2, 3*n//4, n-1]
qt   = [t[i] for i in qs]
qc   = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#EDE7F6']   # soft blues/greens
pclr = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']    # bold text colors

qs_slice = [0, n//4, n//2, 3*n//4, n]

def qstats(ref, plat, sig_for_T):
    out = []
    for i in range(4):
        sl  = slice(qs_slice[i], qs_slice[i+1])
        rr  = np.sqrt(np.mean(ref[sl]**2)); rp = np.sqrt(np.mean(plat[sl]**2))
        red = (1 - rp/rr)*100 if rr > 0 else 0.0
        T   = median_period_seg(sig_for_T[sl], t[sl], min_amp=0.3)
        out.append((rr, rp, red, T))
    return out

r_stats = qstats(r2, r1, r2)
p_stats = qstats(p2, p1, p2)

# ── Overall ───────────────────────────────────────────────────────────────
rms_r2 = np.sqrt(np.mean(r2**2)) or 1e-9; rms_r1 = np.sqrt(np.mean(r1**2))
rms_p2 = np.sqrt(np.mean(p2**2)) or 1e-9; rms_p1 = np.sqrt(np.mean(p1**2))
red_r  = (1 - rms_r1/rms_r2)*100;          red_p  = (1 - rms_p1/rms_p2)*100

# ── Dominant freq overall ─────────────────────────────────────────────────
freqs = rfftfreq(n, d=dt_med)
def dom_f(sig):
    m = np.abs(rfft(sig - sig.mean()))
    return freqs[np.argmax(m[1:]) + 1]
r2_f = dom_f(r2); p2_f = dom_f(p2)

# ── Gains used (from first row if logged, else defaults) ──────────────────
ROLL_GAIN  = 600; PITCH_GAIN = 300
ROLL_KFF   = 70;  PITCH_KFF  = 40
ROLL_KD    = 10;  PITCH_KD   = -5
MAX_D_ERPM = 600

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#f8f9fa')
gs = fig.add_gridspec(4, 2, hspace=0.50, wspace=0.30,
                      height_ratios=[3, 2.5, 2.5, 2.5])

fig.suptitle(
    f"FF+PID TUNE — {os.path.basename(CSV)}\n"
    f"ROLL_GAIN={ROLL_GAIN}  PITCH_GAIN={PITCH_GAIN}  "
    f"ROLL_KFF={ROLL_KFF}  PITCH_KFF={PITCH_KFF}  "
    f"ROLL_KD={ROLL_KD}  PITCH_KD={PITCH_KD}\n"
    f"Roll {red_r:.1f}% reduction  |  Pitch {red_p:.1f}% reduction  |  "
    f"{t[-1]-t[0]:.1f}s test  |  "
    f"Roll dom.freq={r2_f:.3f}Hz  Pitch dom.freq={p2_f:.3f}Hz",
    fontsize=10, fontweight='bold', y=0.995)

# ─────────────────────────────────────────────────────────────────────────
# ROW 0 — Angle panels
# ─────────────────────────────────────────────────────────────────────────
for col, (ref, plat, stats, name, db, red) in enumerate([
        (r2, r1, r_stats, "Roll",  0.5, red_r),
        (p2, p1, p_stats, "Pitch", 1.0, red_p)]):
    ax = fig.add_subplot(gs[0, col])
    for i in range(4):
        ax.axvspan(qt[i], qt[i+1], alpha=0.18, color=qc[i], zorder=0)
    ax.plot(t, ref,  'b-',  alpha=0.55, lw=0.9, label='Boat (IMU2)')
    ax.plot(t, plat, color='#E53935', alpha=0.85, lw=1.1, label='Platform (IMU1)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.fill_between(t, -db, db, alpha=0.07, color='green', label=f'±{db}° deadband')
    ymax = max(np.abs(ref).max(), np.abs(plat).max()) * 1.18 or 1.0

    # Reduction % labels at top of each quarter
    for i in range(4):
        mid = (qt[i] + qt[i+1]) / 2
        ax.text(mid, ymax * 0.91, f"{stats[i][2]:.0f}%",
                ha='center', fontsize=9, color=pclr[i], fontweight='bold')

    # Per-quarter T label at bottom of each quarter
    for i in range(4):
        T = stats[i][3]
        if T is not None:
            mid = (qt[i] + qt[i+1]) / 2
            ax.text(mid, -ymax * 0.88, f"T≈{T:.1f}s",
                    ha='center', va='top', fontsize=8,
                    color='#333', fontstyle='italic')

    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(f"{name} Angle  |  Overall {red:.1f}% reduction", fontsize=10, fontweight='bold')
    ax.set_ylabel("Degrees"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)

# ─────────────────────────────────────────────────────────────────────────
# ROW 1 — Command breakdown (line style)
# ─────────────────────────────────────────────────────────────────────────
for col, (p_t, i_t, ff_t, d_t, cmd, name, kff, kp, kd, dcap) in enumerate([
        (pr, ir, fr, dr, cr, "Roll",  ROLL_KFF,  ROLL_GAIN,  ROLL_KD,  MAX_D_ERPM),
        (pp, ip, fp, dp, cp, "Pitch", PITCH_KFF, PITCH_GAIN, PITCH_KD, MAX_D_ERPM)]):
    ax = fig.add_subplot(gs[1, col])
    for i in range(4):
        ax.axvspan(qt[i], qt[i+1], alpha=0.10, color=qc[i], zorder=0)
    ax.plot(t, ff_t, color='#F59E0B', lw=0.9, alpha=0.9,
            label=f'FF (Kff={kff}, max={np.abs(ff_t).max():.0f})')
    ax.plot(t, d_t,  color='#8B5CF6', lw=0.9, alpha=0.9,
            label=f'D (Kd={kd}, max={np.abs(d_t).max():.0f})')
    ax.plot(t, p_t,  color='#60A5FA', lw=0.9, alpha=0.8,
            label=f'P (Kp={kp})')
    ax.axhline(MAX_D_ERPM,  color='red', lw=0.8, ls=':', alpha=0.6,
               label=f'D cap ±{MAX_D_ERPM}')
    ax.axhline(-MAX_D_ERPM, color='red', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_title(f"{name} Command  (P / FF / D breakdown)", fontsize=9, fontweight='bold')
    ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(t[0], t[-1])

# ─────────────────────────────────────────────────────────────────────────
# ROW 2 — Current  |  |D term|
# ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
for i in range(4):
    ax.axvspan(qt[i], qt[i+1], alpha=0.10, color=qc[i], zorder=0)
ax.plot(t, ra, color='#1976D2', lw=0.9, alpha=0.85, label='Roll')
ax.plot(t, pa, color='#D32F2F', lw=0.9, alpha=0.85, label='Pitch')
ax.axhline(4.5, color='orange', lw=1.0, ls='--', label='4.5A rated')
ax.set_title(f"Motor Current  |  Roll max={ra.max():.2f}A  Pitch max={pa.max():.2f}A",
             fontsize=9, fontweight='bold')
ax.set_ylabel("Amps"); ax.set_xlabel("Time (s)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

ax = fig.add_subplot(gs[2, 1])
for i in range(4):
    ax.axvspan(qt[i], qt[i+1], alpha=0.10, color=qc[i], zorder=0)
ax.plot(t, np.abs(dr), color='#7C3AED', lw=0.9, alpha=0.85,
        label=f'|D roll| (max={np.abs(dr).max():.0f})')
ax.plot(t, np.abs(dp), color='#DB2777', lw=0.9, alpha=0.85,
        label=f'|D pitch| (max={np.abs(dp).max():.0f})')
ax.axhline(MAX_D_ERPM, color='red', lw=1.0, ls='--', alpha=0.7, label=f'D cap {MAX_D_ERPM}')
ax.set_title(f"|D Term| Magnitude  |  Peak {max(np.abs(dr).max(), np.abs(dp).max()):.0f} ERPM",
             fontsize=9, fontweight='bold')
ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

# ─────────────────────────────────────────────────────────────────────────
# ROW 3 — Stabilization by Quarter bar chart (full width)
# ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, :])
x   = np.arange(4); w = 0.35
r_reds = [s[2] for s in r_stats]
p_reds = [s[2] for s in p_stats]
b1 = ax.bar(x - w/2, r_reds, w, label='Roll',  color='#1976D2', alpha=0.85)
b2 = ax.bar(x + w/2, p_reds, w, label='Pitch', color='#D32F2F', alpha=0.85)
for bars, c in [(b1, '#1565C0'), (b2, '#B71C1C')]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=c)

# T labels below each group (roll T below group center-left, pitch T below center-right)
for i in range(4):
    Tr = r_stats[i][3]; Tp = p_stats[i][3]
    label_r = f"T≈{Tr:.1f}s" if Tr is not None else "T=n/a"
    label_p = f"T≈{Tp:.1f}s" if Tp is not None else "T=n/a"
    ax.text(x[i] - w/2, -5, label_r, ha='center', va='top',
            fontsize=8, color='#1565C0', fontstyle='italic')
    ax.text(x[i] + w/2, -5, label_p, ha='center', va='top',
            fontsize=8, color='#B71C1C', fontstyle='italic')

# Phase time labels below x-axis
phase_labels = [f"Q1\n{t[qs[0]]:.0f}–{t[qs[1]]:.0f}s",
                f"Q2\n{t[qs[1]]:.0f}–{t[qs[2]]:.0f}s",
                f"Q3\n{t[qs[2]]:.0f}–{t[qs[3]]:.0f}s",
                f"Q4\n{t[qs[3]]:.0f}–{t[-1]:.0f}s"]
ax.set_xticks(x)
ax.set_xticklabels(phase_labels, fontsize=9)
ax.set_ylabel("% Reduction"); ax.set_ylim(-18, 110)
ax.axhline(0, color='k', lw=0.5)
ax.set_title("Stabilization by Quarter  |  Wave period (T) shown below bars",
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='y')

plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}")
