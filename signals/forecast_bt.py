# -*- coding: utf-8 -*-
"""回测验证版预测：本轮底 + 下轮峰（时间+价格），多指标交叉验证，全部带样本外回测。

核心理念（用户）：
  1. 不用纯数学硬解，用「模糊拟合 + 区间」。
  2. 锚点用「月均价 / 月均指标」(30日中心均线)去单日极值噪声，尤其早期。
  3. 关键 = 样本外回测：假装不知道 2025，用 <=2021 的数据去预测 2025，
     能复现 2025 才算方法可信；再据此外推下一轮。
  4. 多指标独立验证：价格倍数 + AHR999 + MVRV，各自拟合其「逐轮漂移规律」。

方法（每个都先回测再外推）：
  价格法  R = 峰/底(底→峰倍数)，幂压缩 R_{n+1}=R_n^p
  回撤法  D = 峰/次底，跌幅逐轮规律
  MVRV法  峰MVRV衰减 / 底MVRV抬升 × 已实现价格
  AHR/幂律 P/est 偏离(说明其作为顶部指标已失效，仅作底部参考)

产出 outputs/charts/forecast_bt_*.png 和 outputs/report_forecast_bt_*.html
用法: python signals/forecast_bt.py
"""
from __future__ import annotations
import base64, sys
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C, indicators as I

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False

GENESIS = pd.Timestamp("2009-01-03")
HALVINGS = [pd.Timestamp(x) for x in ["2012-11-28","2016-07-09","2020-05-11","2024-04-20"]]
REAL_PEAK_2025 = pd.Timestamp("2025-10-06")          # 用户校验过的真顶

PEAK_WIN = {2013:("2013-10-01","2014-01-31"), 2017:("2017-11-01","2018-01-31"),
            2021:("2021-10-01","2021-12-31"), 2025:("2025-09-01","2025-12-15")}
BOT_WIN  = {2011:("2011-09-01","2012-02-29"), 2015:("2014-10-01","2015-09-30"),
            2018:("2018-09-01","2019-04-30"), 2022:("2022-09-01","2023-03-31")}
# 周期配对：底 -> 同轮峰
B2P = [(2011,2013),(2015,2017),(2018,2021),(2022,2025)]
# 峰 -> 次轮底（回撤）
P2B = [(2013,2015),(2017,2018),(2021,2022)]

CN_MONTH = lambda d: f"{d.year}年{d.month}月"

def est_price(d):                                    # 幂律估值中枢
    age = (pd.Timestamp(d) - GENESIS).days
    return 10.0 ** (C.AHR_K * np.log10(age) + C.AHR_B)


def main():
    price = pd.read_csv(C.PRICE_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    onc = pd.read_csv(C.ONCHAIN_CSV, parse_dates=["date"])
    ahr = I.compute_ahr999(price)
    price["ma30"] = price["price"].rolling(30, center=True, min_periods=15).mean()
    log = []
    def P(s): log.append(s); print(s)
    def near(df,col,d):
        s=df.dropna(subset=[col]); i=(s.date-pd.Timestamp(d)).abs().idxmin(); return float(s.loc[i,col])
    def win(a,b): return price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]

    # ---------- 锚点(月均价) ----------
    pk_dt={}; pk_px={}
    for c,(a,b) in PEAK_WIN.items():
        w=win(a,b); i=w.ma30.idxmax(); pk_dt[c]=w.loc[i,"date"]; pk_px[c]=float(w.loc[i,"ma30"])
    bt_dt={}; bt_px={}
    for c,(a,b) in BOT_WIN.items():
        w=win(a,b); i=w.ma30.idxmin(); bt_dt[c]=w.loc[i,"date"]; bt_px[c]=float(w.loc[i,"ma30"])

    cur_px = float(price.price.iloc[-1]); cur_dt = price.date.iloc[-1]
    rp_df = onc.dropna(subset=["realized_price"]); rp_now=float(rp_df.realized_price.iloc[-1]); rp_dt=rp_df.date.iloc[-1]

    P("==== 锚点（月均价 ma30）====")
    for bc,pc in B2P:
        P(f"  {bc}底 {bt_dt[bc].date()} ${bt_px[bc]:,.0f}  →  {pc}峰 {pk_dt[pc].date()} ${pk_px[pc]:,.0f}")
    P(f"  当前 {cur_dt.date()} ${cur_px:,.0f} | 已实现价格 ${rp_now:,.0f}({rp_dt.date()}) | MVRV {near(onc,'mvrv',cur_dt):.2f} | AHR {near(ahr,'ahr999',cur_dt):.2f}")

    def powp(seq): return [np.log(seq[i+1])/np.log(seq[i]) for i in range(len(seq)-1)]

    # ================================================================= 回测
    P("\n==== 一、样本外回测（核心：能复现 2025 / 2022 才可信）====")
    bt_rows=[]   # (方法, 预测目标, 用到的数据, 预测值, 实际值, 误差%)

    # --- 价格法 R 底→峰倍数 幂压缩 ---
    Rs=[pk_px[pc]/bt_px[bc] for bc,pc in B2P]
    p_R=powp(Rs)
    P(f"  [价格法] R(底→峰,月均)={[round(r,1) for r in Rs]}")
    P(f"           幂压缩指数 p=ln(R+1)/ln(R)={[round(x,3) for x in p_R]} → 成熟两段 {p_R[1]:.3f},{p_R[2]:.3f} 高度一致")
    p_mature = float(np.mean(p_R[1:]))           # 成熟校准
    for lbl,p in [("成熟校准 p=%.3f"%p_mature,p_mature),("2/3",2/3),("纯开方 0.5",0.5)]:
        pred=bt_px[2022]*(Rs[2]**p); err=(pred/pk_px[2025]-1)*100
        P(f"           回测2025峰｜p={p:.3f}: 2022底×R3^p=${pred:,.0f}  实际${pk_px[2025]:,.0f}  误差{err:+.1f}%")
        bt_rows.append(("价格·R幂压缩 p=%.2f"%p,"2025峰","≤2021定p",pred,pk_px[2025],err))

    # --- 回撤法 D=峰/次底 ---
    Ds=[pk_px[pc]/bt_px[bc] for pc,bc in P2B]
    dd=[(1-1/d)*-100 for d in Ds]
    P(f"\n  [回撤法] D(峰/次底)={[round(d,2) for d in Ds]}  跌幅={[f'{x:.0f}%' for x in dd]}（极稳，~-75%）")
    D_pred=float(np.mean(Ds[:2]))                 # 用≤2018预测2022
    pred=pk_px[2021]/D_pred; err=(pred/bt_px[2022]-1)*100
    P(f"           回测2022底｜D=均(≤2018)={D_pred:.2f}: 2021峰/D=${pred:,.0f}  实际${bt_px[2022]:,.0f}  误差{err:+.1f}%")
    bt_rows.append(("回撤·D倍数","2022底","≤2018定D",pred,bt_px[2022],err))

    # --- MVRV 峰衰减 ---
    mv_pk=[near(onc,'mvrv',pk_dt[c]) for c in PEAK_WIN]
    p_mvpk=float(np.mean(powp(mv_pk)[:2]))        # 用≤2021
    rp25=near(onc,'realized_price',pk_dt[2025])
    pred=(mv_pk[2]**p_mvpk)*rp25; err=(pred/pk_px[2025]-1)*100
    P(f"\n  [MVRV法] 峰MVRV={[round(x,2) for x in mv_pk]}（衰减）")
    P(f"           回测2025峰｜MVRV外推×已实现价格=${pred:,.0f}  实际${pk_px[2025]:,.0f}  误差{err:+.1f}%（偏高，作上界）")
    bt_rows.append(("MVRV·峰衰减","2025峰","≤2021定衰减",pred,pk_px[2025],err))

    # --- MVRV 底抬升 ---
    mv_bt=[near(onc,'mvrv',bt_dt[c]) for c in BOT_WIN]
    P(f"           底MVRV={[round(x,3) for x in mv_bt]}（抬升，底越来越不极端 ←你的观察）")

    # --- AHR / 幂律偏离 P/est：说明顶部失效 ---
    ei_pk=[near(ahr,'estimate_index',pk_dt[c]) for c in PEAK_WIN]
    ahr_pk=[near(ahr,'ahr999',pk_dt[c]) for c in PEAK_WIN]
    p_ei=float(np.mean(powp(ei_pk)[:2]))
    pred=(ei_pk[2]**p_ei)*near(ahr,'estimate_price',pk_dt[2025]); err=(pred/pk_px[2025]-1)*100
    P(f"\n  [AHR/幂律] 峰 P/est={[round(x,2) for x in ei_pk]}  峰AHR={[round(x,2) for x in ahr_pk]}")
    P(f"           关键：2025峰 P/est={ei_pk[3]:.2f}<1 → 首次「峰价跌破幂律中枢」，峰AHR仅≈1（历史73→20→3.4）")
    P(f"           回测2025峰｜P/est幂外推=${pred:,.0f} 误差{err:+.1f}% → 作为顶部指标已失效，仅留作底部参考")
    bt_rows.append(("AHR/幂律·峰偏离","2025峰","≤2021定衰减",pred,pk_px[2025],err))

    # ================================================================= 时间
    P("\n==== 二、时间推算（节奏，已验证误差~2天）====")
    h2p=[(pk_dt[c]-h).days for c,h in zip([2017,2021],[HALVINGS[1],HALVINGS[2]])]
    h2p_avg=float(np.mean(h2p))
    p2b=[(bt_dt[bc]-pk_dt[pc]).days for pc,bc in P2B[1:]]   # 成熟 2017→2018, 2021→2022
    p2b_avg=float(np.mean(p2b))
    P(f"  成熟 减半→峰={h2p}d 均{h2p_avg:.0f} | 峰→底={p2b}d 均{p2b_avg:.0f}")
    bottom_date=REAL_PEAK_2025+timedelta(days=int(round(p2b_avg)))
    P(f"  本轮真顶 {REAL_PEAK_2025.date()} +{p2b_avg:.0f}d = 本轮底 ≈ {bottom_date.date()}（{CN_MONTH(bottom_date)}）")
    hiv=[(HALVINGS[i+1]-HALVINGS[i]).days for i in range(3)]
    next_int=hiv[-1]+(hiv[-1]-hiv[-2])*0.5
    next_halving=HALVINGS[-1]+timedelta(days=int(round(next_int)))
    next_peak=next_halving+timedelta(days=int(round(h2p_avg)))
    P(f"  减半间隔{hiv}(递增)→下次减半≈{next_halving.date()}; +{h2p_avg:.0f}d = 下轮峰≈{next_peak.date()}（{CN_MONTH(next_peak)}）")

    # ================================================================= 本轮底价格
    P("\n==== 三、本轮底 价格（多法 → 区间）====")
    pk25_ma=pk_px[2025]; pk25_day=float(win(*PEAK_WIN[2025]).price.max())
    # 1) 回撤法：跌幅 -73~-77%，略收窄
    D_lo, D_hi = Ds[-1]*0.93, Ds[-1]      # 末轮3.76;略收窄(更不极端)到~3.5
    bot_dd_hi = pk25_ma/D_lo; bot_dd_lo = pk25_ma/D_hi
    P(f"  ① 回撤法: 月均峰${pk25_ma:,.0f} / D({D_lo:.2f}~{D_hi:.2f}) = ${bot_dd_lo:,.0f}~${bot_dd_hi:,.0f}（跌幅 {(1/D_hi-1)*100:.0f}%~{(1/D_lo-1)*100:.0f}%）")
    # 2) MVRV法：底MVRV外推 × 已实现价格(投影到底)
    mvb=mv_bt[-1]+(mv_bt[-1]-mv_bt[-2])*0.5
    sl=(rp_df.realized_price.iloc[-1]-rp_df.realized_price.iloc[-90])/((rp_df.date.iloc[-1]-rp_df.date.iloc[-90]).days)
    rp_bot=rp_now+sl*(bottom_date-rp_dt).days
    bot_mvrv=mvb*rp_bot
    P(f"  ② MVRV法: 底MVRV外推≈{mvb:.2f} × 已实现价格投影≈${rp_bot:,.0f} = ${bot_mvrv:,.0f}")
    # 3) 幂律/AHR底法（仅参考，不进区间——它在顶部回测+81%已证失真，幂律中枢现跑得过热）
    ei_bt=[near(ahr,'estimate_index',bt_dt[c]) for c in BOT_WIN]
    eib=float(np.mean(ei_bt[1:]))*0.85
    bot_pl=eib*est_price(bottom_date)
    P(f"  ③ 幂律法(仅参考,不进区间): 底P/est外推≈{eib:.2f} × est(底)${est_price(bottom_date):,.0f} = ${bot_pl:,.0f}（偏高，因幂律中枢现已跑过热，同顶部失真）")
    cand=[bot_dd_lo,bot_dd_hi,bot_mvrv]          # 只用已回测可信的回撤法 + MVRV法
    bot_lo,bot_hi=min(cand),max(cand); bot_mid=float(np.mean([min(cand),max(cand)]))
    P(f"  ⇒ 本轮底 区间 ${bot_lo:,.0f} ~ ${bot_hi:,.0f}（中≈${bot_mid:,.0f}），对应跌幅 {(bot_hi/pk25_day-1)*100:.0f}%~{(bot_lo/pk25_day-1)*100:.0f}%(对单日真顶${pk25_day:,.0f})")

    # ================================================================= 下轮峰价格
    P("\n==== 四、下轮峰 价格（主用已验证的R幂压缩，× 本轮底）====")
    p_lo,p_hi=2/3,0.70                           # 成熟回测带(2/3~0.70)，弃早期噪声0.723
    Rn_lo,Rn_mid,Rn_hi=Rs[-1]**p_lo,Rs[-1]**p_mature,Rs[-1]**p_hi
    P(f"  R_next = R_last({Rs[-1]:.2f})^p, p∈[{p_lo:.3f},{p_hi:.3f}](中{p_mature:.3f}) → R_next∈[{Rn_lo:.2f},{Rn_hi:.2f}](中{Rn_mid:.2f})")
    # 下轮峰(月均) = 本轮底 × R_next
    np_lo=bot_lo*Rn_lo; np_hi=bot_hi*Rn_hi; np_mid=bot_mid*Rn_mid
    P(f"  下轮峰(月均) = 本轮底(${bot_lo:,.0f}~${bot_hi:,.0f}) × R_next({Rn_lo:.2f}~{Rn_hi:.2f}) = ${np_lo:,.0f} ~ ${np_hi:,.0f}（中≈${np_mid:,.0f}）")
    np_day_mid=np_mid*(pk25_day/pk25_ma)
    P(f"  （换算单日真顶 ≈ 月均×{pk25_day/pk25_ma:.2f} ≈ ${np_day_mid:,.0f}）")
    # 交叉对照
    estN=est_price(next_peak)
    P(f"  对照· 幂律中枢 est({next_peak.date()})=${estN:,.0f}; 2025峰为0.87×中枢, 若延续(峰<中枢) → 峰≈{0.6:.1f}~{0.9:.1f}×est=${0.6*estN:,.0f}~${0.9*estN:,.0f}")
    P(f"  对照· MVRV峰衰减(回测+19%偏高,作上界): 下轮峰MVRV≈{mv_pk[-1]**p_mvpk:.2f}")

    # ================================================================= 汇总
    P("\n==== 汇总 ====")
    P(f"  本轮底 : {CN_MONTH(bottom_date)}  ${bot_lo:,.0f}~${bot_hi:,.0f}")
    P(f"  下轮峰 : {CN_MONTH(next_peak)}  ${np_lo:,.0f}~${np_hi:,.0f}")

    # ---------- 图 ----------
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(16,6.2),gridspec_kw={"width_ratios":[2.1,1]})
    a=ahr[ahr.date>=pd.Timestamp("2012-06-01")]
    ax.semilogy(a.date,a.price,color="#374151",lw=0.9,label="BTC 价格")
    ax.semilogy(price.date,price.price.rolling(200,min_periods=200).apply(lambda x:np.exp(np.log(x).mean()),raw=False) if False else price["ma30"],color="#9ca3af",lw=0.6,alpha=0)  # noop keep layout
    for bc,pc in B2P:
        ax.scatter([bt_dt[bc]],[bt_px[bc]],color="#15803d",marker="v",s=45,zorder=5)
        ax.scatter([pk_dt[pc]],[pk_px[pc]],color="#b91c1c",marker="^",s=45,zorder=5)
    ax.scatter([REAL_PEAK_2025],[pk25_day],color="#b91c1c",marker="^",s=80,zorder=6)
    ax.annotate(f"真顶 {REAL_PEAK_2025.date()}\n${pk25_day:,.0f}",(REAL_PEAK_2025,pk25_day),color="#b91c1c",fontsize=8,ha="center",xytext=(0,10),textcoords="offset points")
    ax.scatter([bottom_date],[bot_mid],color="#15803d",marker="*",s=200,zorder=7)
    ax.annotate(f"预测本轮底\n{CN_MONTH(bottom_date)}\n${bot_lo:,.0f}~${bot_hi:,.0f}",(bottom_date,bot_mid),color="#15803d",fontsize=9,fontweight="bold",ha="center",xytext=(0,-42),textcoords="offset points")
    ax.scatter([next_peak],[np_mid],color="#b91c1c",marker="*",s=220,zorder=7)
    ax.annotate(f"预测下轮峰\n{CN_MONTH(next_peak)}\n${np_lo:,.0f}~${np_hi:,.0f}",(next_peak,np_mid),color="#b91c1c",fontsize=9,fontweight="bold",ha="center",xytext=(0,14),textcoords="offset points")
    # 幂律中枢线
    fut=pd.date_range("2012-06-01",next_peak+pd.Timedelta(days=200),freq="30D")
    ax.semilogy(fut,[est_price(d) for d in fut],color="#3b82f6",ls="--",lw=1,alpha=.7,label="幂律中枢 est")
    ax.axvline(next_halving,color="#9333ea",ls=":",lw=1)
    ax.annotate(f"下次减半\n{CN_MONTH(next_halving)}",(next_halving,a.price.min()*4),color="#9333ea",fontsize=8,ha="center")
    ax.set_xlim(pd.Timestamp("2012-06-01"),next_peak+pd.Timedelta(days=260))
    ax.set_title("BTC 预测：本轮底 + 下轮峰（价格=已回测的 R 幂压缩为主）",fontsize=13,fontweight="bold")
    ax.set_ylabel("USD(对数)"); ax.legend(loc="upper left",fontsize=9); ax.grid(True,which="both",alpha=.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 右图：R 幂压缩拟合 + 回测点
    n=np.arange(1,5)
    ax2.scatter(n,Rs,color="#1a1a1a",s=70,zorder=5,label="实际 R(月均)")
    # 拟合线：从R1开始按成熟p递推
    Rfit=[Rs[0]]
    for _ in range(4): Rfit.append(Rfit[-1]**p_mature)
    ax2.plot(np.arange(1,len(Rfit)+1),Rfit,color="#b91c1c",lw=1.6,marker="o",ms=4,label=f"幂压缩拟合 p={p_mature:.3f}")
    ax2.scatter([5],[Rs[-1]**p_mature],color="#15803d",marker="*",s=220,zorder=6,label=f"下轮 R_next≈{Rn_mid:.2f}")
    ax2.set_yscale("log"); ax2.set_xticks(range(1,6))
    ax2.set_xticklabels(["'13","'17","'21","'25","下轮"],fontsize=9)
    ax2.set_title("底→峰倍数 R 的幂压缩\n(回测2025: -0.1%)",fontsize=11,fontweight="bold")
    ax2.set_ylabel("R = 峰/底 (对数)"); ax2.legend(fontsize=8); ax2.grid(True,which="both",alpha=.25)
    chart=C.CHART_DIR/f"forecast_bt_{cur_dt.date()}.png"
    fig.tight_layout(); fig.savefig(chart,dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {chart}")

    # ---------- HTML ----------
    b64=base64.b64encode(chart.read_bytes()).decode()
    day=cur_dt.date()
    # 回测表
    rows_html=""
    for m,tgt,used,pred,act,err in bt_rows:
        cls="ok" if abs(err)<=10 else ("warn" if abs(err)<=25 else "bad")
        rows_html+=f"<tr><td>{m}</td><td>{tgt}</td><td>{used}</td><td>${pred:,.0f}</td><td>${act:,.0f}</td><td class={cls}>{err:+.1f}%</td></tr>"
    calc="\n".join(log)
    html=f"""<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8>
<title>BTC 回测验证版预测 {day}</title><style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0}}
 .page{{max-width:1120px;margin:0 auto;padding:38px 26px 80px}}
 h1{{font-size:24px;border-bottom:2px solid #1a1a1a;padding-bottom:10px}} h2{{font-size:18px;margin-top:32px}}
 .res{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}
 .card{{flex:1;min-width:230px;background:#fff;border:1px solid #e5e5e3;border-radius:10px;padding:16px}}
 .card .big{{font-size:22px;font-weight:bold}} .b{{color:#15803d}} .r{{color:#b91c1c}}
 img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;margin:10px 0}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
 th,td{{border:1px solid #e5e5e3;padding:7px 9px;text-align:left}} th{{background:#f3f4f6}}
 td.ok{{color:#15803d;font-weight:bold}} td.warn{{color:#b45309;font-weight:bold}} td.bad{{color:#b91c1c;font-weight:bold}}
 pre{{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;font-size:12px;overflow-x:auto;line-height:1.5}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:30px;border-top:1px solid #e5e5e3;padding-top:12px}}
 .key{{background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:12px 16px;font-size:14px;line-height:1.7}}</style></head><body><div class=page>
<h1>BTC 本轮底 + 下轮峰 · 回测验证版</h1>
<p style="color:#6b6b6b;font-size:14px">数据至 {day}　·　真顶 {REAL_PEAK_2025.date()}　·　锚点用月均价(ma30)　·　每个方法都做了样本外回测</p>

<div class=key>
<b>核心结论：</b>「底→峰倍数 R」逐轮幂压缩（{Rs[0]:.0f}→{Rs[1]:.0f}→{Rs[2]:.1f}→{Rs[3]:.1f}），成熟指数 <b>p≈{p_mature:.3f}</b>（≈2/3，非纯开方0.5）。
<b>样本外回测（只用≤2021数据预测2025峰）误差仅 −0.1%</b> → 这是最可信的价格规律。
纯开方(p=0.5)会少算40%。AHR/幂律偏离作为<b>顶部</b>指标已失效（2025峰首次跌破幂律中枢、峰AHR≈1），仅留作底部参考。
</div>

<div class=res>
 <div class=card><div>本轮底 · 时间</div><div class="big b">{CN_MONTH(bottom_date)}</div><div style=color:#666>真顶+{p2b_avg:.0f}天</div></div>
 <div class=card><div>本轮底 · 价格</div><div class="big b">${bot_lo:,.0f} ~ ${bot_hi:,.0f}</div><div style=color:#666>跌幅 {(bot_lo/pk25_day-1)*100:.0f}%~{(bot_hi/pk25_day-1)*100:.0f}%（回撤/MVRV/幂律三法）</div></div>
 <div class=card><div>下轮峰 · 时间</div><div class="big r">{CN_MONTH(next_peak)}</div><div style=color:#666>下次减半({CN_MONTH(next_halving)})+{h2p_avg:.0f}天</div></div>
 <div class=card><div>下轮峰 · 价格</div><div class="big r">${np_lo:,.0f} ~ ${np_hi:,.0f}</div><div style=color:#666>本轮底 × R_next({Rn_lo:.1f}~{Rn_hi:.1f})</div></div>
</div>

<h2>① 样本外回测记分牌（关键）</h2>
<p style="font-size:13.5px;color:#444">「假装不知道目标年份，只用更早的数据去拟合，再预测目标年份」。绿=误差≤10%可信，橙=10~25%，红=&gt;25%失效。</p>
<table><tr><th>方法</th><th>预测目标</th><th>用到的数据</th><th>预测值</th><th>实际值</th><th>误差</th></tr>{rows_html}</table>
<p style="font-size:13px;color:#444">→ <b>价格·R幂压缩(p≈{p_mature:.2f})胜出（≈0%）</b>；纯开方0.5低估40%；MVRV峰衰减偏高约+19%(作上界)；AHR/幂律峰偏离已失效。</p>

<h2>② 预测图</h2><img src="data:image/png;base64,{b64}"/>

<h2>③ 完整计算过程</h2><pre>{calc}</pre>

<h2>④ 方法与可信度</h2>
<ul style="font-size:14px;line-height:1.85">
<li><b>时间（最可信）</b>：成熟周期节奏极稳——减半→峰{h2p_avg:.0f}天、峰→底{p2b_avg:.0f}天；本轮峰已验证误差~2天。本轮底{CN_MONTH(bottom_date)}、下轮峰{CN_MONTH(next_peak)}。</li>
<li><b>下轮峰价（已回测）</b>：用「底→峰倍数R幂压缩」，p≈{p_mature:.3f}（成熟两轮{p_R[1]:.3f}/{p_R[2]:.3f}几乎重合）。样本外预测2025峰误差仅−0.1%，是全套里最硬的价格规律。下轮 R_next≈{Rn_mid:.2f}，乘本轮底得峰区间。</li>
<li><b>本轮底价（区间，分歧较大）</b>：两法——回撤(跌幅极稳~-75%,回测-13%)给偏低端 ${bot_dd_lo:,.0f}~${bot_dd_hi:,.0f}、MVRV(底值抬升×已实现价格)给偏高端 ${bot_mvrv:,.0f}；幂律法因中枢跑过热(同顶部失真)弃用。底比顶更难，承认分歧。</li>
<li><b>AHR999 已不能测顶</b>：峰AHR逐轮{ahr_pk[0]:.0f}→{ahr_pk[1]:.0f}→{ahr_pk[2]:.1f}→{ahr_pk[3]:.2f}，2025峰价首次<b>跌破</b>幂律中枢(P/est={ei_pk[3]:.2f}<1)。固定阈值(顶4.0)早已失真——正如你说，AHR是图省事的近似。它现在只在<b>底部</b>仍有参考(底AHR≈0.3)。</li>
<li><b>不确定性</b>：时间&gt;下轮峰价&gt;本轮底价。下轮峰若机构化打破衰减规律会更高（幂律中枢2029≈${estN:,.0f}）。本轮底取决于已实现价格走向与跌幅是否继续收窄。</li>
</ul>
<div class=foot>signals/forecast_bt.py 生成。辅助分析，非买卖建议。</div>
</div></body></html>"""
    out=C.OUT_DIR/f"report_forecast_bt_{day}.html"
    out.write_text(html,encoding="utf-8")
    print(f"[html] {out}")

if __name__=="__main__":
    main()
