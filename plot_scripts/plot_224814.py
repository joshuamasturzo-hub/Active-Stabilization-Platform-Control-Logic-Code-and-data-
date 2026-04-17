import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260406_224814.csv"
OUT = "fpd_tune_20260406_224814_analysis.png"

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

# Quarter boundaries
qs = [0, n//4, n//2, 3*n//4, n]
qt = [t[min(i, n-1)] for i in qs]

def quarter_stats(sig_ref, sig_plat):
    stats = []
    for i in range(4):
        sl = slice(qs[i], qs[i+1])
        rr = np.sqrt(np.mean(sig_ref[sl]**2))
        rp = np.sqrt(np.mean(sig_plat[sl]**2))
        red = (1-rp/rr)*100 if rr > 0 else 0
        stats.append((rr, rp, red, t[qs[i]], t[min(qs[i+1]-1, n-1)]))
    return stats

# P contribution = total - D - FF
pr = cr - dr - fr
pp = cp - dp - fp

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
r_stats = quarter_stats(r2, r1)
p_stats = quarter_stats(p2, p1)

# wave period per quarter (dom freq of IMU2)
def quarter_period(sig):
    periods = []
    for i in range(4):
        sl = slice(qs[i], qs[i+1])
        s = sig[sl]
        if len(s) < 10: periods.append(0); continue
        f = rfftfreq(len(s), d=dt)
        m = np.abs(rfft(s - s.mean()))
        pf = f[np.argmax(m[1:])+1] if np.argmax(m[1:])+1 < len(f) else 0.1
        periods.append(1/pf if pf > 0 else 0)
    return periods

roll_periods  = quarter_period(r2)
pitch_periods = quarter_period(p2)

colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#f8f9fa')
gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

# Read gains for title
try:
    import json
    with open('gains.json') as gf: g = json.load(gf)
    title = (f"FF+PD TUNE — fpd_tune_data_20260406_224814\n"
             f"PITCH_GAIN={g.get('PITCH_GAIN','?')} PITCH_KD={g.get('PITCH_KD','?')} "
             f"MAX_D_ERPM={g.get('MAX_D_ERPM','?')}  "
             f"Best result yet: Roll {red_r:.1f}% | Pitch {red_p:.1f}%")
except:
    title = f"FF+PD TUNE — 20260406_224814  |  Roll {red_r:.1f}%  Pitch {red_p:.1f}%"
fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)

qc = ['#FFF9C4','#E8F5E9','#FFF3E0','#F3E5F5']

# ── Row 0: Roll angle | Pitch angle ──────────────────────────────────────────
for col, (sig_r, sig_p, lbl, stats, pname, red) in enumerate([
    (r2, r1, "Roll", r_stats, "Roll", red_r),
    (p2, p1, "Pitch", p_stats, "Pitch", red_p),
]):
    ax = fig.add_subplot(gs[0, col])
    for i in range(4):
        ax.axvspan(qt[i], qt[i+1], alpha=0.15, color=qc[i], zorder=0)
        mid = (qt[i]+qt[i+1])/2
        ax.text(mid, ax.get_ylim()[1] if False else 0, f"{stats[i][2]:.0f}%",
                ha='center', va='bottom', fontsize=8, color=colors[i], fontweight='bold')
    ax.plot(t, sig_r, 'b-', alpha=0.6, lw=0.9, label='IMU2 (ref)')
    ax.plot(t, sig_p, 'r-', alpha=0.85, lw=1.1, label='Platform (IMU1)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    db = 0.5 if col==0 else 1.0
    ax.fill_between(t, -db, db, alpha=0.08, color='green', label=f'±{db}° deadband')
    ax.set_title(f"{pname} Angle  |  Overall {red:.1f}% reduction", fontsize=10, fontweight='bold')
    ax.set_ylabel("Degrees"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.25)
    ax.set_xlim(t[0], t[-1])
    # annotate quarters
    for i in range(4):
        mid = (qt[i]+qt[i+1])/2
        ypos = ax.get_ylim()[0]*0.85 if ax.get_ylim()[0] < 0 else ax.get_ylim()[1]*0.85
        ax.text(mid, ax.get_ylim()[1]*0.88, f"{stats[i][2]:.0f}%",
                ha='center', fontsize=9, color=colors[i], fontweight='bold')
        period_val = roll_periods[i] if col==0 else pitch_periods[i]
        ax.text(mid, ax.get_ylim()[0]*0.88 if ax.get_ylim()[0]<0 else ax.get_ylim()[1]*0.08,
                f"T={period_val:.1f}s", ha='center', fontsize=7, color='gray')

# ── Row 1: Command breakdown ──────────────────────────────────────────────────
for col, (p_term, ff_term, d_term, cmd, lbl) in enumerate([
    (pr, fr, dr, cr, "Roll"),
    (pp, fp, dp, cp, "Pitch"),
]):
    ax = fig.add_subplot(gs[1, col])
    ax.fill_between(t, 0, p_term,  alpha=0.35, color='#1976D2', label=f'P (max={np.abs(p_term).max():.0f})')
    ax.fill_between(t, 0, ff_term, alpha=0.35, color='#388E3C', label=f'FF (max={np.abs(ff_term).max():.0f})')
    ax.fill_between(t, 0, d_term,  alpha=0.35, color='#F57C00', label=f'D (max={np.abs(d_term).max():.0f})')
    ax.plot(t, cmd, 'k-', lw=0.8, alpha=0.7, label=f'Total cmd')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_title(f"{lbl} Command  (P / FF / D breakdown)  —  D never hit cap", fontsize=9, fontweight='bold')
    ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, loc='upper right', ncol=2); ax.grid(True, alpha=0.25)
    ax.set_xlim(t[0], t[-1])

# ── Row 2: Amps | D term magnitude ───────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, ra, 'b-', lw=0.9, alpha=0.85, label='Roll')
ax.plot(t, pa, 'r-', lw=0.9, alpha=0.85, label='Pitch')
ax.axhline(4.5, color='red', lw=1.0, ls=':', label='4.5A rated')
ax.set_title(f"Motor Current  |  Roll max={ra.max():.2f}A  Pitch max={pa.max():.2f}A", fontsize=9, fontweight='bold')
ax.set_ylabel("Amps"); ax.set_xlabel("Time (s)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

ax = fig.add_subplot(gs[2, 1])
ax.plot(t, np.abs(dr), 'b-', lw=0.9, alpha=0.85, label=f'D roll (max={np.abs(dr).max():.0f})')
ax.plot(t, np.abs(dp), 'r-', lw=0.9, alpha=0.85, label=f'D pitch (max={np.abs(dp).max():.0f})')
try:
    import json
    with open('gains.json') as gf: g = json.load(gf)
    cap = g.get('MAX_D_ERPM', 600)
except: cap = 600
ax.axhline(cap, color='k', lw=0.8, ls=':', label=f'D cap {cap:.0f}')
ax.set_title(f"|D Term| Magnitude  —  well within cap (max {max(np.abs(dr).max(),np.abs(dp).max()):.0f})", fontsize=9, fontweight='bold')
ax.set_ylabel("ERPM"); ax.set_xlabel("Time (s)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xlim(t[0], t[-1])

# ── Row 3: Phase bar chart ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, :])
x = np.arange(4); w = 0.35
phase_labels = [f'Q{i+1}\nT≈{roll_periods[i]:.1f}s' for i in range(4)]
r_reds = [s[2] for s in r_stats]
p_reds = [s[2] for s in p_stats]
b1 = ax.bar(x - w/2, r_reds, w, label='Roll', color='#1976D2', alpha=0.85)
b2 = ax.bar(x + w/2, p_reds, w, label='Pitch', color='#D32F2F', alpha=0.85)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1976D2')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#D32F2F')
ax.set_xticks(x); ax.set_xticklabels(phase_labels, fontsize=9)
ax.set_ylabel("% Reduction"); ax.set_ylim(min(0, min(r_reds+p_reds))-10, 110)
ax.axhline(0, color='k', lw=0.5)
ax.set_title("Stabilization by Phase  |  Wave period (T) shown below each bar", fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='y')

plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}")
