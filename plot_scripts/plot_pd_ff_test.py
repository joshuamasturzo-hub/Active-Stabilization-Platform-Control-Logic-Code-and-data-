import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "pd_control_data_20260403_184426.csv"
OUT = "dry test 14 PD+FF — Roll600 D25 FF300 Pitch300 D8 FF150 OSCILLATION.png"

df = pd.read_csv(CSV)
t=df["t_s"].values; r1=df["imu1_roll_deg"].values; p1=df["imu1_pitch_deg"].values
r2=df["imu2_roll_deg"].values; p2=df["imu2_pitch_deg"].values
cr=df["cmd_roll_erpm"].values; cp=df["cmd_pitch_erpm"].values
ra=df["motor_roll_amps"].values; pa=df["motor_pitch_amps"].values
fr=df["ff_roll_erpm"].values; fp=df["ff_pitch_erpm"].values

n=len(t); dt=np.median(np.diff(t))
rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100

fig,axes=plt.subplots(4,1,figsize=(14,14))
fig.suptitle("Dry Test 14 — PD+FF | ISSUE: IMU2 DC offset causes constant FF push → oscillation",
             fontsize=12,fontweight="bold",y=0.99)

ax=axes[0]
ax.plot(t,r2,"b-",alpha=0.6,lw=1.0,label="IMU2 Reference")
ax.plot(t,r1,"r-",alpha=0.9,lw=1.2,label="IMU1 Platform")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.set_ylabel("Roll (deg)",fontsize=10)
ax.set_title(f"Roll | Reduction={red_r:.1f}% — degraded by FF fighting P term",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[1]
ax.plot(t,p2,"b-",alpha=0.6,lw=1.0,label="IMU2 Reference")
ax.plot(t,p1,"r-",alpha=0.9,lw=1.2,label="IMU1 Platform")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.set_ylabel("Pitch (deg)",fontsize=10)
ax.set_title(f"Pitch | Reduction={red_p:.1f}% | IMU2 mean={p2.mean():.2f}° (DC offset) → FF always on",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[2]; ax2=ax.twinx()
ax.plot(t,cr,"g-",alpha=0.8,lw=1.0,label="Roll cmd"); ax.plot(t,cp,"m-",alpha=0.8,lw=1.0,label="Pitch cmd")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax2.plot(t,ra,"g--",alpha=0.6,lw=0.9,label="Roll A"); ax2.plot(t,pa,"m--",alpha=0.6,lw=0.9,label="Pitch A")
ax2.axhline(4.0,color="red",lw=0.8,ls=":",label="4A limit")
ax.set_ylabel("ERPM"); ax2.set_ylabel("Amps")
ax.set_title(f"Commands | Roll {ra.max():.2f}A  Pitch {pa.max():.2f}A",fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,labs1+labs2,fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[3]
ax.plot(t,fr,"g-",lw=1.2,label=f"FF roll (max={np.abs(fr).max():.0f} mean={np.abs(fr).mean():.0f} ERPM)")
ax.plot(t,fp,"m-",lw=1.2,label=f"FF pitch (max={np.abs(fp).max():.0f} mean={np.abs(fp).mean():.0f} ERPM)")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.set_ylabel("FF term (ERPM)"); ax.set_xlabel("Time (s)")
ax.set_title(f"Feedforward — pitch FF mean={np.abs(fp).mean():.0f} ERPM constantly on due to IMU2 DC offset ({p2.mean():.1f}°)",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

for ax in axes: ax.set_xlim(t[0],t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT,dpi=150,bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
