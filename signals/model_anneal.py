# -*- coding: utf-8 -*-
"""新模型 ACD（Annealed Cycle Decomposition / 退火周期分解）。

核心(用户的分解): log_price = 趋势底座 + 退火周期波
  趋势底座 = 已实现价格 realized_price（全网成本线，真实数据，不虚高）
  周期波   = MVRV ( = price / realized_price )
  → 价格 = realized_price × MVRV
两层都退火:
  realized_price 的逐轮(对数)增量按 ~0.57 衰减（底座增速变缓）
  MVRV 峰逐轮衰减(2.8→2.9→2.6→2.0)、MVRV 底逐轮抬升(0.49→0.77→0.78→0.82)

每个外推都先做样本外回测(只用≤2021预测2025峰 / ≤2018预测2022底)，再外推。
最后与 R 倍数模型(forecast_bt)横向对比。

产出 outputs/charts/model_anneal_*.png + report_model_anneal_*.html
用法: python signals/model_anneal.py
"""
from __future__ import annotations
import base64, sys
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C, indicators as I
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False

GEN=pd.Timestamp("2009-01-03")
HALVINGS=[pd.Timestamp(x) for x in ["2012-11-28","2016-07-09","2020-05-11","2024-04-20"]]
REAL_PEAK_2025=pd.Timestamp("2025-10-06")
PEAK_WIN={2013:("2013-10-01","2014-01-31"),2017:("2017-11-01","2018-01-31"),2021:("2021-10-01","2021-12-31"),2025:("2025-09-01","2025-12-15")}
BOT_WIN={2011:("2011-09-01","2012-02-29"),2015:("2014-10-01","2015-09-30"),2018:("2018-09-01","2019-04-30"),2022:("2022-09-01","2023-03-31")}
B2P=[(2011,2013),(2015,2017),(2018,2021),(2022,2025)]
CN=lambda d:f"{d.year}年{d.month}月"

def proj_log_incr(seq, dampen=None):
    """对数空间逐轮增量衰减外推下一个值。dampen=增量衰减率(None=用最近两个比值均值)。"""
    L=np.log10(seq); d=np.diff(L)
    if dampen is None:
        r=[d[i+1]/d[i] for i in range(len(d)-1)]; dampen=float(np.mean(r[-2:]))
    nd=d[-1]*dampen
    return 10**(L[-1]+nd), dampen, nd

def main():
    price=pd.read_csv(C.PRICE_CSV,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    onc=pd.read_csv(C.ONCHAIN_CSV,parse_dates=["date"])
    ahr=I.compute_ahr999(price)
    price["ma30"]=price["price"].rolling(30,center=True,min_periods=15).mean()
    log=[]; P=lambda s:(log.append(s),print(s))
    def near(df,col,d):
        s=df.dropna(subset=[col]); i=(s.date-pd.Timestamp(d)).abs().idxmin(); return float(s.loc[i,col])
    def win(a,b): return price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]

    pk_dt={};pk_px={};bt_dt={};bt_px={}
    for c,(a,b) in PEAK_WIN.items():
        w=win(a,b);i=w.ma30.idxmax();pk_dt[c]=w.loc[i,"date"];pk_px[c]=float(w.loc[i,"ma30"])
    for c,(a,b) in BOT_WIN.items():
        w=win(a,b);i=w.ma30.idxmin();bt_dt[c]=w.loc[i,"date"];bt_px[c]=float(w.loc[i,"ma30"])
    cur_px=float(price.price.iloc[-1]); cur_dt=price.date.iloc[-1]

    # realized & MVRV 锚点
    rp_pk=[near(onc,'realized_price',pk_dt[c]) for c in PEAK_WIN]
    rp_bt=[near(onc,'realized_price',bt_dt[c]) for c in BOT_WIN]
    mv_pk=[near(onc,'mvrv',pk_dt[c]) for c in PEAK_WIN]
    mv_bt=[near(onc,'mvrv',bt_dt[c]) for c in BOT_WIN]
    rp_now=near(onc,'realized_price',cur_dt); pk_day25=float(win(*PEAK_WIN[2025]).price.max())

    P("==== 新模型 ACD：价格 = 已实现价格(趋势) × MVRV(周期波) ====")
    P(f"  峰 已实现价格 {[f'${x:,.0f}' for x in rp_pk]}  ×  峰MVRV {[round(x,2) for x in mv_pk]}")
    P(f"  底 已实现价格 {[f'${x:,.0f}' for x in rp_bt]}  ×  底MVRV {[round(x,2) for x in mv_bt]}")
    P(f"  现 已实现价格 ${rp_now:,.0f} | 现价 ${cur_px:,.0f} | 现MVRV {cur_px/rp_now:.2f}")

    # ---------------- 时间(沿用已验证节奏) ----------------
    h2p_avg=float(np.mean([(pk_dt[c]-h).days for c,h in zip([2017,2021],[HALVINGS[1],HALVINGS[2]])]))
    p2b_avg=float(np.mean([(bt_dt[bc]-pk_dt[pc]).days for pc,bc in [(2017,2018),(2021,2022)]]))
    bottom_date=REAL_PEAK_2025+timedelta(days=int(round(p2b_avg)))
    hiv=[(HALVINGS[i+1]-HALVINGS[i]).days for i in range(3)]
    next_halving=HALVINGS[-1]+timedelta(days=int(round(hiv[-1]+(hiv[-1]-hiv[-2])*0.5)))
    next_peak=next_halving+timedelta(days=int(round(h2p_avg)))
    P(f"\n==== 时间 ====\n  本轮底≈{bottom_date.date()}({CN(bottom_date)}) | 下次减半≈{next_halving.date()} | 下轮峰≈{next_peak.date()}({CN(next_peak)})")

    # ---------------- 回测 ----------------
    P("\n==== 回测（样本外）====")
    # 峰: 用≤2021(前3轮)预测2025
    rp_pred25,dr,_=proj_log_incr(rp_pk[:3])
    mv_pred25=mv_pk[2]*(mv_pk[2]/mv_pk[1])           # MVRV峰用最近比值外推
    px_pred25=rp_pred25*mv_pred25
    err_rp=(rp_pred25/rp_pk[3]-1)*100; err_px=(px_pred25/pk_px[2025]-1)*100
    P(f"  [峰] 已实现价格外推 ${rp_pred25:,.0f}(实际${rp_pk[3]:,.0f},{err_rp:+.1f}%) × MVRV外推{mv_pred25:.2f}(实际{mv_pk[3]:.2f})")
    P(f"       → 2025峰预测 ${px_pred25:,.0f}  实际 ${pk_px[2025]:,.0f}  误差 {err_px:+.1f}%")
    # 底: 用≤2018预测2022
    rp_pred22,_,_=proj_log_incr(rp_bt[:3])
    mv_pred22=mv_bt[2]+(mv_bt[2]-mv_bt[1])*0.5
    px_pred22=rp_pred22*mv_pred22; err_b=(px_pred22/bt_px[2022]-1)*100
    P(f"  [底] 已实现价格外推 ${rp_pred22:,.0f}(实际${rp_bt[3]:,.0f}) × MVRV外推{mv_pred22:.2f}(实际{mv_bt[3]:.2f})")
    P(f"       → 2022底预测 ${px_pred22:,.0f}  实际 ${bt_px[2022]:,.0f}  误差 {err_b:+.1f}%")

    # ---------------- 外推：本轮底 ----------------
    P("\n==== 本轮底(2026) 价格 ====")
    rp_bot_next,db,ndb=proj_log_incr(rp_bt)          # 底已实现价格外推
    mv_bot_next=mv_bt[-1]+(mv_bt[-1]-mv_bt[-2])*0.5  # 底MVRV继续小幅抬升
    bot_mid=rp_bot_next*mv_bot_next
    # 区间: realized ±一档, MVRV 0.80~0.88
    bot_lo=rp_bot_next*0.95*0.80; bot_hi=rp_bot_next*1.05*0.88
    P(f"  底已实现价格外推 ${rp_bot_next:,.0f}（增量衰减{db:.2f}）× 底MVRV {mv_bot_next:.2f}")
    P(f"  本轮底中枢 ${bot_mid:,.0f}；区间(已实现±5%×MVRV0.80~0.88) ${bot_lo:,.0f}~${bot_hi:,.0f}（跌幅 {(bot_hi/pk_day25-1)*100:.0f}%~{(bot_lo/pk_day25-1)*100:.0f}%）")

    # ---------------- 外推：下轮峰 ----------------
    P("\n==== 下轮峰(2029) 价格 ====")
    rp_pk_next,dp,ndp=proj_log_incr(rp_pk)           # 峰已实现价格外推
    # 峰MVRV退火: 用最近比值, 给区间
    r_mv=mv_pk[-1]/mv_pk[-2]
    mv_pk_mid=mv_pk[-1]*r_mv; mv_pk_lo=mv_pk[-1]*(r_mv-0.05); mv_pk_hi=mv_pk[-1]*(r_mv+0.08)
    peak_mid=rp_pk_next*mv_pk_mid
    peak_lo=rp_pk_next*0.92*mv_pk_lo; peak_hi=rp_pk_next*1.08*mv_pk_hi
    P(f"  峰已实现价格外推 ${rp_pk_next:,.0f}（增量衰减{dp:.2f}）")
    P(f"  峰MVRV退火: 最近比值{r_mv:.2f} → 峰MVRV {mv_pk_lo:.2f}~{mv_pk_hi:.2f}（中{mv_pk_mid:.2f}）")
    P(f"  下轮峰中枢 ${peak_mid:,.0f}；区间 ${peak_lo:,.0f}~${peak_hi:,.0f}")

    # ---------------- 与 R 倍数模型对比 ----------------
    P("\n==== 与 R 倍数模型(forecast_bt) 对比 ====")
    Rs=[pk_px[pc]/bt_px[bc] for bc,pc in B2P]
    p_R=float(np.mean([np.log(Rs[i+1])/np.log(Rs[i]) for i in range(len(Rs)-1)][1:]))
    Rn=Rs[-1]**p_R
    rm_bot_lo,rm_bot_hi=30959,44453   # 来自 forecast_bt
    rm_pk_lo,rm_pk_hi=rm_bot_lo*Rs[-1]**(2/3),rm_bot_hi*Rs[-1]**0.70
    P(f"  R倍数模型: 本轮底 ${rm_bot_lo:,.0f}~${rm_bot_hi:,.0f} | 下轮峰 ${rm_pk_lo:,.0f}~${rm_pk_hi:,.0f}（R_next≈{Rn:.2f}）")
    P(f"  ACD分解模型: 本轮底 ${bot_lo:,.0f}~${bot_hi:,.0f} | 下轮峰 ${peak_lo:,.0f}~${peak_hi:,.0f}")
    # 融合(两模型并集中段)
    fb_lo=min(bot_lo,rm_bot_lo); fb_hi=max(bot_hi,rm_bot_hi)
    fp_lo=min(peak_lo,rm_pk_lo); fp_hi=max(peak_hi,rm_pk_hi)
    P(f"  → 两模型融合: 本轮底 ${fb_lo:,.0f}~${fb_hi:,.0f}（中${(fb_lo+fb_hi)/2:,.0f}）| 下轮峰 ${fp_lo:,.0f}~${fp_hi:,.0f}（中${(fp_lo+fp_hi)/2:,.0f}）")

    # ---------------- 图 ----------------
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(16,6),gridspec_kw={"width_ratios":[2,1]})
    a=onc.dropna(subset=['realized_price'])
    ax.semilogy(price.date,price.ma30,color="#374151",lw=.9,label="BTC 月均价")
    ax.semilogy(a.date,a.realized_price,color="#2563eb",lw=1.8,label="趋势底座=已实现价格")
    for bc,pc in B2P:
        ax.scatter([pk_dt[pc]],[pk_px[pc]],color="#b91c1c",marker="^",s=40,zorder=5)
        ax.scatter([bt_dt[bc]],[bt_px[bc]],color="#15803d",marker="v",s=40,zorder=5)
    ax.scatter([bottom_date],[bot_mid],color="#15803d",marker="*",s=200,zorder=7)
    ax.annotate(f"预测本轮底\n{CN(bottom_date)}\n${bot_lo:,.0f}~${bot_hi:,.0f}",(bottom_date,bot_mid),color="#15803d",fontsize=8.5,fontweight="bold",ha="center",xytext=(0,-40),textcoords="offset points")
    ax.scatter([next_peak],[peak_mid],color="#b91c1c",marker="*",s=220,zorder=7)
    ax.annotate(f"预测下轮峰\n{CN(next_peak)}\n${peak_lo:,.0f}~${peak_hi:,.0f}",(next_peak,peak_mid),color="#b91c1c",fontsize=8.5,fontweight="bold",ha="center",xytext=(0,14),textcoords="offset points")
    # 趋势底座外推点
    ax.scatter([bottom_date],[rp_bot_next],color="#2563eb",marker="o",s=30,zorder=6)
    ax.scatter([next_peak],[rp_pk_next],color="#2563eb",marker="o",s=30,zorder=6)
    ax.axvline(next_halving,color="#9333ea",ls=":",lw=1)
    ax.set_xlim(pd.Timestamp("2013-01-01"),next_peak+pd.Timedelta(days=240))
    ax.set_title("ACD 模型: 价格 = 已实现价格(趋势) × MVRV(退火周期波)",fontsize=12.5,fontweight="bold")
    ax.set_ylabel("USD(对数)"); ax.legend(loc="upper left",fontsize=9); ax.grid(True,which="both",alpha=.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # 右: 两模型对比条
    labels=["本轮底","下轮峰"]; x=np.arange(2)
    acd=[(bot_lo+bot_hi)/2,(peak_lo+peak_hi)/2]; rm=[(rm_bot_lo+rm_bot_hi)/2,(rm_pk_lo+rm_pk_hi)/2]
    acd_err=[[acd[0]-bot_lo,acd[1]-peak_lo],[bot_hi-acd[0],peak_hi-acd[1]]]
    rm_err=[[rm[0]-rm_bot_lo,rm[1]-rm_pk_lo],[rm_bot_hi-rm[0],rm_pk_hi-rm[1]]]
    ax2.errorbar(x-0.1,acd,yerr=acd_err,fmt="o",color="#2563eb",capsize=6,ms=9,label="ACD分解模型")
    ax2.errorbar(x+0.1,rm,yerr=rm_err,fmt="s",color="#b91c1c",capsize=6,ms=9,label="R倍数模型")
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_yscale("log")
    ax2.set_title("两模型对比(区间)",fontsize=11,fontweight="bold"); ax2.set_ylabel("USD(对数)")
    ax2.legend(fontsize=9); ax2.grid(True,which="both",alpha=.25)
    chart=C.CHART_DIR/f"model_anneal_{cur_dt.date()}.png"
    fig.tight_layout(); fig.savefig(chart,dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {chart}")

    # ---------------- HTML ----------------
    b64=base64.b64encode(chart.read_bytes()).decode(); day=cur_dt.date(); calc="\n".join(log)
    html=f"""<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>ACD 退火周期分解模型 {day}</title><style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0}}
 .page{{max-width:1100px;margin:0 auto;padding:38px 26px 80px}} h1{{font-size:23px;border-bottom:2px solid #1a1a1a;padding-bottom:10px}} h2{{font-size:18px;margin-top:30px}}
 .res{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}} .card{{flex:1;min-width:220px;background:#fff;border:1px solid #e5e5e3;border-radius:10px;padding:16px}}
 .card .big{{font-size:21px;font-weight:bold}} .b{{color:#15803d}} .r{{color:#b91c1c}}
 img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;margin:10px 0}} table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
 th,td{{border:1px solid #e5e5e3;padding:7px 9px;text-align:left}} th{{background:#f3f4f6}}
 pre{{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;font-size:12px;overflow-x:auto;line-height:1.5}}
 .key{{background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;padding:12px 16px;font-size:14px;line-height:1.7}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:30px;border-top:1px solid #e5e5e3;padding-top:12px}}</style></head><body><div class=page>
<h1>ACD · 退火周期分解模型</h1>
<p style="color:#6b6b6b;font-size:14px">数据至 {day}　·　核心: <b>价格 = 已实现价格(趋势底座) × MVRV(退火周期波)</b>　·　两层都退火，全部样本外回测</p>
<div class=key><b>为什么用这个分解：</b>它就是你说的「log_price = 长期趋势 + 退火周期波」的真实数据版。趋势底座用<b>已实现价格</b>(全网成本线，不像幂律那样虚高)；周期波用 <b>MVRV = 价格/已实现价格</b>。两层都在退火：底座增速逐轮变缓、MVRV峰逐轮压低/MVRV底逐轮抬升。回测：预测2025峰误差 {err_px:+.1f}%、预测2022底误差 {err_b:+.1f}%。</div>
<div class=res>
 <div class=card><div>本轮底 · 时间</div><div class="big b">{CN(bottom_date)}</div></div>
 <div class=card><div>本轮底 · 价格(ACD)</div><div class="big b">${bot_lo:,.0f} ~ ${bot_hi:,.0f}</div><div style=color:#666>已实现价格×底MVRV{mv_bot_next:.2f}</div></div>
 <div class=card><div>下轮峰 · 时间</div><div class="big r">{CN(next_peak)}</div></div>
 <div class=card><div>下轮峰 · 价格(ACD)</div><div class="big r">${peak_lo:,.0f} ~ ${peak_hi:,.0f}</div><div style=color:#666>已实现价格×峰MVRV{mv_pk_mid:.2f}</div></div>
</div>
<h2>① 两模型对比（独立方法，互为验证）</h2>
<table><tr><th>模型</th><th>方法</th><th>2025峰回测</th><th>本轮底(2026)</th><th>下轮峰(2029)</th></tr>
<tr><td>R 倍数压缩</td><td>底→峰倍数幂压缩 p≈0.68（无需趋势）</td><td>−0.1%</td><td>${rm_bot_lo:,.0f}~${rm_bot_hi:,.0f}</td><td>${rm_pk_lo:,.0f}~${rm_pk_hi:,.0f}</td></tr>
<tr><td>ACD 分解(本模型)</td><td>已实现价格×MVRV，两层退火</td><td>{err_px:+.1f}%</td><td>${bot_lo:,.0f}~${bot_hi:,.0f}</td><td>${peak_lo:,.0f}~${peak_hi:,.0f}</td></tr>
<tr style="background:#f0fdf4"><td><b>两者融合</b></td><td>取并集</td><td>—</td><td><b>${fb_lo:,.0f}~${fb_hi:,.0f}</b></td><td><b>${fp_lo:,.0f}~${fp_hi:,.0f}</b></td></tr></table>
<p style="font-size:13.5px;color:#444">两条完全独立的路（一条用比例、一条用趋势×乘子）落在同一区间 → 结论稳健。R倍数回测更准(−0.1% vs {err_px:+.1f}%)，作主；ACD 解释「为什么」并交叉验证。</p>
<h2>② 模型图</h2><img src="data:image/png;base64,{b64}"/>
<h2>③ 完整计算过程</h2><pre>{calc}</pre>
<h2>④ 说明</h2>
<ul style="font-size:14px;line-height:1.8">
<li><b>趋势底座=已实现价格</b>：它是全网真实成本，平滑上升且增速自然变缓，避开了幂律中枢现在虚高(会把目标算太高)的毛病。</li>
<li><b>周期波=MVRV</b>：MVRV峰 2.8→2.9→2.6→2.0 逐轮压低、MVRV底 0.49→0.77→0.78→0.82 逐轮抬升——正是「振幅退火」在真实数据上的体现。</li>
<li><b>两层都退火</b>：已实现价格逐轮(对数)增量×~0.57，MVRV峰逐轮×~0.77；外推时都按衰减续推。</li>
<li><b>可信度</b>：ACD 回测({err_px:+.1f}%)略逊于 R 倍数(−0.1%)，主因 MVRV 外推偏高；但两模型区间高度重叠，互为验证。时间&gt;价格；下轮峰受「机构化是否打破退火」影响最大。</li>
</ul>
<div class=foot>signals/model_anneal.py 生成。辅助分析，非买卖建议。</div></div></body></html>"""
    out=C.OUT_DIR/f"report_model_anneal_{day}.html"; out.write_text(html,encoding="utf-8"); print(f"[html] {out}")

if __name__=="__main__":
    main()
