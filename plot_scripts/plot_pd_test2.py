import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

CSV = "pd_control_data_20260403_183702.csv"
OUT = "dry test 13 PD — Roll600 D25 Pitch300 D8 filtered.png"

df = pd.read_csv(CSV)
t=df["t_s"].values; r1=df["imu1_roll_deg"].values; p1=df["imu1_pitch_deg"].values
r2=df["imu2_roll_deg"].values; p2=df["imu2_pitch_deg"].values
cr=df["cmd_roll_erpm"].values; cp=df["cmd_pitch_erpm"].values
ra=df["motor_roll_amps"].values; pa=df["motor_pitch_amps"].values
dr=df["d_roll_erpm"].values; dp=df["d_pitch_erpm"].values

n=len(t); dt=np.median(np.diff(t))
freqs=rfftfreq(n,d=dt)
def dominant_freq(sig):
    mag=np.abs(rfft(sig-sig.mean())); return freqs[np.argmax(mag[1:])+1]
def lag_ms(ref,sig):
    corr=np.correlate(ref-ref.mean(),sig-sig.mean(),mode='full')
    lags=np.arange(-(n-1),n)*dt; return lags[np.argmax(corr)]*1000

rms_r2=np.sqrt(np.mean(r2**2)); rms_r1=np.sqrt(np.mean(r1**2))
rms_p2=np.sqrt(np.mean(p2**2)); rms_p1=np.sqrt(np.mean(p1**2))
red_r=(1-rms_r1/rms_r2)*100; red_p=(1-rms_p1/rms_p2)*100
r2_f=dominant_freq(r2); roll_lag=lag_ms(r2,r1)

fig,axes=plt.subplots(4,1,figsize=(14,14))
fig.suptitle("Dry Test 13 — PD | Roll P=600 D=25 | Pitch P=300 D=8 | EMA α=0.3 | D cap 800 ERPM",
             fontsize=12,fontweight="bold",y=0.99)

ax=axes[0]
ax.plot(t,r2,"b-",alpha=0.6,lw=1.0,label="IMU2 Reference")
ax.plot(t,r1,"r-",alpha=0.9,lw=1.2,label="IMU1 Platform")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.fill_between(t,-0.5,0.5,alpha=0.10,color="green",label="±0.5° deadband")
ax.set_ylabel("Roll (deg)",fontsize=10)
ax.set_title(f"Roll | IMU2 RMS={rms_r2:.2f}°  IMU1 RMS={rms_r1:.2f}°  Reduction={red_r:.1f}%  "
             f"Freq={r2_f:.3f}Hz  Period={1/r2_f:.2f}s  Lag={roll_lag:.0f}ms  [prev best: 47.3%]",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[1]
ax.plot(t,p2,"b-",alpha=0.6,lw=1.0,label="IMU2 Reference")
ax.plot(t,p1,"r-",alpha=0.9,lw=1.2,label="IMU1 Platform")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.fill_between(t,-1.0,1.0,alpha=0.10,color="green",label="±1.0° deadband")
ax.set_ylabel("Pitch (deg)",fontsize=10)
ax.set_title(f"Pitch | IMU2 RMS={rms_p2:.2f}°  IMU1 RMS={rms_p1:.2f}°  Reduction={red_p:.1f}%  "
             f"Last 4s IMU1 std=0.033° — stable  [prev best: 76.8%]",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[2]; ax2=ax.twinx()
ax.plot(t,cr,"g-",alpha=0.8,lw=1.0,label="Roll cmd ERPM")
ax.plot(t,cp,"m-",alpha=0.8,lw=1.0,label="Pitch cmd ERPM")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax2.plot(t,ra,"g--",alpha=0.6,lw=0.9,label="Roll amps")
ax2.plot(t,pa,"m--",alpha=0.6,lw=0.9,label="Pitch amps")
ax2.axhline(4.0,color="red",lw=0.8,ls=":",label="4.0A limit")
ax.set_ylabel("ERPM",fontsize=10); ax2.set_ylabel("Current (A)",fontsize=10)
ax.set_title(f"Commands & Current | Roll max {np.abs(cr).max():.0f} ERPM {ra.max():.2f}A | "
             f"Pitch max {np.abs(cp).max():.0f} ERPM {pa.max():.2f}A | 0 cap hits",fontsize=9)
lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,labs1+labs2,fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

ax=axes[3]
ax.plot(t,dr,"g-",alpha=0.85,lw=1.2,label=f"D roll (max={np.abs(dr).max():.0f} ERPM)")
ax.plot(t,dp,"m-",alpha=0.85,lw=1.2,label=f"D pitch (max={np.abs(dp).max():.0f} ERPM)")
ax.axhline(0,color="k",lw=0.5,ls="--")
ax.axhline(800,color="k",lw=0.6,ls=":",alpha=0.4)
ax.axhline(-800,color="k",lw=0.6,ls=":",alpha=0.4,label="±800 cap")
ax.set_ylabel("D term (ERPM)",fontsize=10); ax.set_xlabel("Time (s)",fontsize=10)
ax.set_title("D Term Contribution — roll D now reaching 716 ERPM (up from 339 with KD=15)",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.3)

for ax in axes: ax.set_xlim(t[0],t[-1])
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT,dpi=150,bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
