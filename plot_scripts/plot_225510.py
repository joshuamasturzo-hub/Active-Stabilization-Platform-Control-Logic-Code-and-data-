import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import rfft, rfftfreq

CSV = "fpd_tune_data_20260406_225510.csv"
OUT = "fpd_tune_20260406_225510_analysis.png"

df = pd.read_csv(CSV)
t   = df["t_s"].values
r1  = df["imu1_roll_deg"].values
p1  = df["imu1_pitch_deg"].values
r2  = df["imu2_roll_deg"].values
p2  = df["imu2_pitch_deg"].values
cr  = df["cmd_roll_erpm"].values
cp  = df["cmd_pitch_erpm"].values
dr  = df["d_roll_erpm"].values
dp  = df["d_pitch_erpm"].values
ffr = df["ff_roll_erpm"].values
ffp = df["ff_pitch_erpm"].values
p_roll  = cr - ffr - dr
p_pitch = cp - ffp - dp
ra  = df["motor_roll_amps"].values
pa  = df["motor_pitch_amps"].values
n   = len(t); dt_med = np.median(np.diff(t))

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
cap_r=(np.abs(cr)>=4999).sum(); cap_p=(np.abs(cp)>=4999).sum()

# 2s window stats for annotation
win = int(2.0/dt_med)
windows = []
for i in range(0, n-win, win):
    sl=slice(i,i+win)
    rr2=np.sqrt(np.mean(r2[sl]**2)); rr1=np.sqrt(np.mean(r1[sl]**2))
    rp2=np.sqrt(np.mean(p2[sl]**2)); rp1=np.sqrt(np.mean(p1[sl]**2))
    red_rw=(1-rr1/rr2)*100 if rr2>0.5 else None
    red_pw=(1-rp1/rp2)*100 if rp2>0.5 else None
    caps=(np.abs(cr[sl])>=4999).sum()+(np.abs(cp[sl])>=4999).sum()
    tmid=t[sl][len(t[sl])//2]
    windows.append((tmid, red_rw, red_pw, caps))

# phase segments
segs=[("Slow\n0-8s",   slice(0,int(8/dt_med))),
      ("Fast\n8-18s",  slice(int(8/dt_med),int(18/dt_med))),
      ("Wind-down\n18-26s", slice(int(18/dt_med),None))]
seg_stats=[]
for label,sl in segs:
    r2s=r2[sl];r1s=r1[sl];p2s=p2[sl];p1s=p1[sl]
    rr2=np.sqrt(np.mean(r2s**2)) if len(r2s) else 0
    rr1=np.sqrt(np.mean(r1s**2)) if len(r1s) else 0
    rp2=np.sqrt(np.mean(p2s**2)) if len(p2s) else 0
    rp1=np.sqrt(np.mean(p1s**2)) if len(p1s) else 0
    seg_stats.append((label,(1-rr1/rr2)*100 if rr2>0 else 0,(1-rp1/rp2)*100 if rp2>0 else 0))

fig=plt.figure(figsize=(16,15))
gs =gridspec.GridSpec(3,2,figure=fig,hspace=0.44,wspace=0.35,top=0.91,bottom=0.28)
gs2=gridspec.GridSpec(1,1,figure=fig,top=0.23,bottom=0.04,left=0.08,right=0.92)

fig.suptitle(
    f"FF+PD TUNE — fpd_tune_data_20260406_225510\n"
    f"Roll {red_r:.1f}% | Pitch {red_p:.1f}% | "
    f"Cap hits: Roll {cap_r} Pitch {cap_p} | Roll amps max {ra.max():.2f}A | D was OFF this run\n"
    f"Unsaturated windows show strong correction — FF gain too high causing saturation in fast sections",
    fontsize=10, fontweight="bold", y=0.99)

def annotate_windows(ax, axis, ypos, windows):
    for tmid, wr, wp, caps in windows:
        val = wr if axis=='roll' else wp
        if val is None: continue
        sat = caps > 20
        color = "gray" if sat else ("green" if val>30 else ("orange" if val>0 else "red"))
        style = "italic" if sat else "normal"
        ax.text(tmid, ypos, f"{'~' if sat else ''}{val:.0f}%",
                ha="center", va="top", fontsize=7.5, color=color,
                fontweight="bold", fontstyle=style)

ylim_r = max(np.abs(r2).max(), np.abs(r1).max()) * 1.15
ylim_p = max(np.abs(p2).max(), np.abs(p1).max()) * 1.15

# Roll angle
ax=fig.add_subplot(gs[0,0])
ax.plot(t,r2,"b-",alpha=0.55,lw=1.0,label="IMU2 (ref)")
ax.plot(t,r1,"r-",alpha=0.9, lw=1.1,label="IMU1 (platform)")
ax.axhline(0,color="k",lw=0.5,ls="--")
# shade saturated zones
sat_mask = np.abs(cr)>=4999
ax.fill_between(t, -ylim_r, ylim_r, where=sat_mask, alpha=0.08, color="red", label="Saturated")
annotate_windows(ax, 'roll', ylim_r*0.95, windows)
ax.set_ylim(-ylim_r, ylim_r); ax.set_ylabel("Degrees",fontsize=9)
ax.set_title(f"Roll Angle | Overall {red_r:.1f}% (gray=saturated window)",fontsize=9)
ax.legend(fontsize=7,loc="lower right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# Pitch angle
ax=fig.add_subplot(gs[0,1])
ax.plot(t,p2,"b-",alpha=0.55,lw=1.0,label="IMU2 (ref)")
ax.plot(t,p1,"r-",alpha=0.9, lw=1.1,label="IMU1 (platform)")
ax.axhline(0,color="k",lw=0.5,ls="--")
sat_mask_p = np.abs(cp)>=4999
ax.fill_between(t,-ylim_p,ylim_p,where=sat_mask_p,alpha=0.08,color="red",label="Saturated")
annotate_windows(ax,'pitch',ylim_p*0.95,windows)
ax.set_ylim(-ylim_p,ylim_p); ax.set_ylabel("Degrees",fontsize=9)
ax.set_title(f"Pitch Angle | Overall {red_p:.1f}% (gray=saturated window)",fontsize=9)
ax.legend(fontsize=7,loc="lower right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# Roll command breakdown
ax=fig.add_subplot(gs[1,0])
ax.fill_between(t,0,p_roll, alpha=0.35,color="blue",  label=f"P (max={np.abs(p_roll).max():.0f})")
ax.fill_between(t,0,ffr,    alpha=0.35,color="orange",label=f"FF (max={np.abs(ffr).max():.0f})")
ax.fill_between(t,0,dr,     alpha=0.5, color="green", label=f"D (max={np.abs(dr).max():.0f})")
ax.plot(t,cr,"k-",lw=0.8,alpha=0.7,label="Total cmd")
ax.axhline(5000, color="red",lw=0.7,ls=":",alpha=0.7)
ax.axhline(-5000,color="red",lw=0.7,ls=":",alpha=0.7,label="±5000 cap")
ax.set_ylabel("ERPM",fontsize=9)
ax.set_title(f"Roll Command (P/FF/D) | Cap hits: {cap_r}",fontsize=9)
ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# Pitch command breakdown
ax=fig.add_subplot(gs[1,1])
ax.fill_between(t,0,p_pitch,alpha=0.35,color="blue",  label=f"P (max={np.abs(p_pitch).max():.0f})")
ax.fill_between(t,0,ffp,    alpha=0.35,color="orange",label=f"FF (max={np.abs(ffp).max():.0f})")
ax.fill_between(t,0,dp,     alpha=0.5, color="green", label=f"D (max={np.abs(dp).max():.0f})")
ax.plot(t,cp,"k-",lw=0.8,alpha=0.7,label="Total cmd")
ax.axhline(5000, color="red",lw=0.7,ls=":",alpha=0.7)
ax.axhline(-5000,color="red",lw=0.7,ls=":",alpha=0.7,label="±5000 cap")
ax.set_ylabel("ERPM",fontsize=9)
ax.set_title(f"Pitch Command (P/FF/D) | Cap hits: {cap_p}",fontsize=9)
ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# Current
ax=fig.add_subplot(gs[2,0])
ax.plot(t,ra,"b-",lw=0.9,alpha=0.85,label="Roll")
ax.plot(t,pa,"r-",lw=0.9,alpha=0.85,label="Pitch")
ax.axhline(4.5,color="red",  lw=1.0,ls="--",label="4.5A rated")
ax.axhline(4.0,color="orange",lw=0.8,ls=":", label="4.0A soft limit")
ax.set_ylabel("Amps",fontsize=9); ax.set_xlabel("Time (s)",fontsize=9)
ax.set_title(f"Motor Current | Roll max={ra.max():.2f}A  Pitch max={pa.max():.2f}A",fontsize=9)
ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# D term
ax=fig.add_subplot(gs[2,1])
ax.plot(t,np.abs(dr),"b-",lw=0.9,alpha=0.85,label=f"|D roll|")
ax.plot(t,np.abs(dp),"r-",lw=0.9,alpha=0.85,label=f"|D pitch|")
ax.axhline(1500,color="k",lw=0.8,ls=":",alpha=0.5,label="cap=1500")
ax.set_ylabel("ERPM",fontsize=9); ax.set_xlabel("Time (s)",fontsize=9)
ax.set_title("|D Term| — D was OFF this run (zero throughout)",fontsize=9)
ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.3); ax.set_xlim(t[0],t[-1])

# Phase bar chart
ax=fig.add_subplot(gs2[0])
labels=[s[0] for s in seg_stats]
rp=[s[1] for s in seg_stats]; pp=[s[2] for s in seg_stats]
x=np.arange(len(labels)); w=0.35
br=ax.bar(x-w/2,rp,w,label="Roll", color="#1f77b4",alpha=0.8)
bp=ax.bar(x+w/2,pp,w,label="Pitch",color="#ff7f0e",alpha=0.8)
for bar in list(br)+list(bp):
    h=bar.get_height()
    c="green" if h>30 else ("orange" if h>0 else "red")
    ax.text(bar.get_x()+bar.get_width()/2,h+1,f"{h:.0f}%",
            ha="center",va="bottom",fontsize=9,fontweight="bold",color=c)
ax.axhline(0,color="k",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=10)
ax.set_ylabel("% Reduction",fontsize=10)
ax.set_title("Stabilization by Phase | Fast section heavily saturated — FF gain too high",fontsize=10)
ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")
ax.set_ylim(min(min(rp),min(pp))-15, max(max(rp),max(pp))+15)

plt.savefig(OUT,dpi=150,bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
