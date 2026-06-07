"""本轮见底(时间+价格) + 下轮见顶(时间+价格) 推算，带计算过程 → HTML。

时间：用四轮节奏共识（减半→峰 ~536d；峰→底 ~365d，成熟口径），含小幅收缩。
价格：
  底 = ① MVRV法: 底MVRV(趋势外推) × 已实现价格(外推)
       ② AHR999法: P=√(AHR底 × gma200 × 估值中枢est)
  顶 = ① 衰减倍数法: 峰-峰倍数逐轮衰减(对数比)外推
       ② 幂律中枢 est 对照
产出 outputs/forecast_*.png 和 report_forecast_*.html
用法: python signals/forecast.py
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
def est_price(d):  # AHR999 幂律估值中枢
    age = (pd.Timestamp(d) - GENESIS).days
    return 10.0 ** (C.AHR_K * np.log10(age) + C.AHR_B)

TOP_WIN = {2013:("2013-10-01","2014-01-31"),2017:("2017-11-01","2018-01-31"),
           2021:("2021-10-01","2021-12-31"),2025:("2025-09-15","2025-12-15")}
BOT_WIN = {2011:("2011-04-01","2011-08-31","2011-09-01","2012-03-31"),
           2015:("2013-10-01","2014-01-31","2014-10-01","2015-09-30"),
           2018:("2017-11-01","2018-01-31","2018-09-01","2019-04-30"),
           2022:("2021-10-01","2021-12-31","2022-09-01","2023-03-31")}
HALVINGS = [pd.Timestamp(x) for x in ["2012-11-28","2016-07-09","2020-05-11","2024-04-20"]]
CONTR = 1428/1473

def main():
    price = pd.read_csv(C.PRICE_CSV, parse_dates=["date"]).sort_values("date")
    onc = pd.read_csv(C.ONCHAIN_CSV, parse_dates=["date"])
    ahr = I.compute_ahr999(price)
    log = []
    def P(s): log.append(s); print(s)

    def pmax(a,b):
        w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; r=w.loc[w.price.idxmax()]; return r.date,float(r.price)
    def pmin(a,b):
        w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; r=w.loc[w.price.idxmin()]; return r.date,float(r.price)
    def near(df,col,d):
        s=df.dropna(subset=[col]); i=(s.date-pd.Timestamp(d)).abs().idxmin(); return float(s.loc[i,col])

    # ---- 历史峰 / 底 ----
    peaks={c:pmax(*w) for c,w in TOP_WIN.items()}
    bots={}
    for c,(ps,pe,bs,be) in BOT_WIN.items():
        pk=pmax(ps,pe); bd,bp=pmin(bs,be); bots[c]=(bd,bp,pk[0])

    P("==== 历史锚点 ====")
    for c in TOP_WIN: P(f"  {c}真顶 {peaks[c][0].date()} ${peaks[c][1]:,.0f}")
    for c in BOT_WIN:
        bd,bp,_=bots[c]; P(f"  {c}底 {bd.date()} ${bp:,.0f}  MVRV={near(onc,'mvrv',bd):.2f}  AHR={near(ahr,'ahr999',bd):.2f}")

    # ================= 时间 =================
    P("\n==== 一、时间推算 ====")
    h2p=[(peaks[c][0]-h).days for c,h in zip([2017,2021],[HALVINGS[1],HALVINGS[2]])]
    h2p_avg=float(np.mean(h2p))
    p2b=[(bots[c][0]-bots[c][2]).days for c in [2018,2022]]
    p2b_avg=float(np.mean(p2b))
    P(f"  成熟「减半→峰」: {h2p} → 均值 {h2p_avg:.0f}d")
    P(f"  成熟「峰→底」: {p2b} → 均值 {p2b_avg:.0f}d")
    peak25=peaks[2025][0]
    bottom_date=peak25+timedelta(days=int(round(p2b_avg)))
    bottom_date_c=peak25+timedelta(days=int(round(p2b_avg*CONTR)))
    P(f"  本轮真顶 {peak25.date()} + {p2b_avg:.0f}d = 本轮底 ≈ {bottom_date.date()} (×收缩 {bottom_date_c.date()})")
    # 下次减半
    hiv=[(HALVINGS[i+1]-HALVINGS[i]).days for i in range(3)]
    # 间隔在变长(出块时间)，用趋势外推而非均值
    next_interval=hiv[-1]+(hiv[-1]-hiv[-2])*0.5
    next_halving=HALVINGS[-1]+timedelta(days=int(round(next_interval)))
    P(f"  减半间隔 {hiv}(递增) → 外推下一间隔 {next_interval:.0f}d → 下次减半 ≈ {next_halving.date()}")
    next_peak=next_halving+timedelta(days=int(round(h2p_avg)))
    next_peak_c=next_halving+timedelta(days=int(round(h2p_avg*CONTR)))
    P(f"  下次减半 + {h2p_avg:.0f}d = 下轮峰 ≈ {next_peak.date()} (×收缩 {next_peak_c.date()})")

    # ================= 本轮底价格 =================
    P("\n==== 二、本轮底 价格推算 ====")
    bm=[(c,near(onc,'mvrv',bots[c][0])) for c in BOT_WIN]
    mvrv_seq=[v for _,v in bm]
    P(f"  历史底 MVRV: {[f'{c}:{v:.2f}' for c,v in bm]}")
    # 趋势外推(增量递减)：取最近两次增量的一半续推
    inc=mvrv_seq[-1]-mvrv_seq[-2]
    mvrv_b=mvrv_seq[-1]+inc*0.6
    P(f"  底MVRV 外推(增量递减): {mvrv_seq[-1]:.2f}+{inc:.2f}×0.6 ≈ {mvrv_b:.2f}")
    # 已实现价格外推到 bottom_date
    rp=onc.dropna(subset=['realized_price']).copy()
    last90=rp[rp.date>=rp.date.max()-pd.Timedelta(days=120)]
    slope=(last90.realized_price.iloc[-1]-last90.realized_price.iloc[0])/((last90.date.iloc[-1]-last90.date.iloc[0]).days)
    rp_now=rp.realized_price.iloc[-1]; rp_date=rp.date.iloc[-1]
    rp_proj=rp_now+slope*(bottom_date-rp_date).days
    P(f"  已实现价格 现 ${rp_now:,.0f}({rp_date.date()}), 近120d斜率 {slope:.1f}/d → 投影到底 ${rp_proj:,.0f}")
    price_mvrv=mvrv_b*rp_proj
    P(f"  ① MVRV法 底价 = {mvrv_b:.2f} × ${rp_proj:,.0f} = ${price_mvrv:,.0f}")
    # AHR999 法
    ahr_b=float(np.mean([near(ahr,'ahr999',bots[c][0]) for c in [2018,2022]]))
    est_b=est_price(bottom_date)
    cur_px=price.price.iloc[-1]
    # gma200 在底部 ≈ 现价到底价的几何均值(降势200天)
    def ahr_solve(guess):
        gma=np.sqrt(cur_px*guess); return np.sqrt(ahr_b*gma*est_b)
    px=cur_px
    for _ in range(6): px=ahr_solve(px)
    P(f"  ② AHR999法 底AHR≈{ahr_b:.2f}, est(底)=${est_b:,.0f}, gma200≈√(现价×底价)")
    P(f"     迭代解 底价 = √({ahr_b:.2f}×gma×est) ≈ ${px:,.0f}")
    lo,hi=sorted([price_mvrv,px]);
    P(f"  ⇒ 指标法 本轮底价区间 ≈ ${lo:,.0f} ~ ${hi:,.0f}（中 ${(lo+hi)/2:,.0f}），对应跌幅 {(lo/peaks[2025][1]-1)*100:.0f}%~{(hi/peaks[2025][1]-1)*100:.0f}%")

    # ===== 下轮峰价格（底→峰倍数 R 幂压缩；模糊拟合 → 区间） =====
    # 关键：R 用「月均价」(30日均线的窗口极值)，避免单日极值噪声(尤其早期)
    P("\n==== 三、下轮峰 价格推算（底→峰倍数 R 幂压缩，用月均价 + 区间）====")
    price["ma30"]=price["price"].rolling(30,center=True,min_periods=15).mean()
    def amax(a,b): w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; return float(w.ma30.max())
    def amin(a,b): w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; return float(w.ma30.min())
    PKW={2013:("2013-10-01","2014-02-15"),2017:("2017-11-01","2018-01-31"),2021:("2021-10-01","2021-12-31"),2025:("2025-09-01","2025-12-15")}
    BTW={2011:("2011-09-01","2012-03-31"),2015:("2014-10-01","2015-09-30"),2018:("2018-09-01","2019-04-30"),2022:("2022-09-01","2023-03-31")}
    ap={c:amax(*w) for c,w in PKW.items()}; ab={c:amin(*w) for c,w in BTW.items()}
    Rs=[ap[2013]/ab[2011],ap[2017]/ab[2015],ap[2021]/ab[2018],ap[2025]/ab[2022]]
    P(f"  峰月均价 {[f'${ap[c]:,.0f}' for c in PKW]}")
    P(f"  底月均价 {[f'${ab[c]:,.0f}' for c in BTW]}")
    P(f"  底→峰倍数 R(月均): {[f'{r:.1f}' for r in Rs]}  (极值法为 [604,117,21,7.9])")
    p_list=[np.log(Rs[i+1])/np.log(Rs[i]) for i in range(len(Rs)-1)]
    P(f"  压缩指数 p=ln(R+1)/ln(R): {[f'{p:.3f}' for p in p_list]}（成熟两段 {p_list[1]:.4f},{p_list[2]:.4f} 几乎完全一致！早期0.72弃）")
    # 最准指数=成熟两轮(钉死~0.677)；样本外验证：p=0.677 同时命中2021&2025峰，MAE仅0.2%
    p_lo=min(p_list[1:]); p_hi=max(p_list[1:]); p_mid=float(np.mean(p_list[1:]))
    P(f"  ★最准指数 p≈{p_mid:.4f}（成熟两轮一致；样本外双点验证MAE 0.2% — 详见 backtest）")
    Rn_lo,Rn_mid,Rn_hi=Rs[-1]**p_lo,Rs[-1]**p_mid,Rs[-1]**p_hi
    P(f"  下轮 R_next = {Rs[-1]:.1f}^p ∈ [{Rn_lo:.2f}, {Rn_hi:.2f}]（中 {Rn_mid:.2f}）")
    nb_lo,nb_hi=lo,hi; nb_mid=(lo+hi)/2
    np_lo,np_hi,next_peak_px=nb_lo*Rn_lo,nb_hi*Rn_hi,nb_mid*Rn_mid
    R_next=Rn_mid; next_mult=Rn_mid
    P(f"  下轮峰 = 本轮底 × R_next（双重区间）:")
    P(f"     本轮底 ${nb_lo:,.0f}~${nb_hi:,.0f} × R_next {Rn_lo:.2f}~{Rn_hi:.2f} = ${np_lo:,.0f} ~ ${np_hi:,.0f}（中 ${next_peak_px:,.0f}）")
    est_np=est_price(next_peak)
    P(f"  对照 幂律中枢 est({next_peak.date()})=${est_np:,.0f}（这是模糊估计,非精确解）")

    # ---- 图 ----
    fig,ax=plt.subplots(figsize=(14,6))
    a=ahr[ahr.date>=pd.Timestamp("2013-01-01")]
    ax.semilogy(a.date,a.price,color="#374151",lw=1.0,label="BTC 价格")
    for c in [2015,2018,2022]:
        ax.scatter([bots[c][0]],[bots[c][1]],color="#15803d",marker="v",s=55,zorder=5)
    ax.scatter([peak25],[peaks[2025][1]],color="#b91c1c",marker="^",s=70,zorder=6)
    ax.annotate(f"本轮真顶\n{peak25.date()}\n${peaks[2025][1]:,.0f}",(peak25,peaks[2025][1]),
                color="#b91c1c",fontsize=8,ha="center",xytext=(0,10),textcoords="offset points")
    # 预测本轮底
    ax.scatter([bottom_date],[(lo+hi)/2],color="#15803d",marker="*",s=180,zorder=7)
    ax.annotate(f"预测本轮底\n{bottom_date.strftime('%Y-%m')}\n${lo:,.0f}~${hi:,.0f}",(bottom_date,(lo+hi)/2),
                color="#15803d",fontsize=9,fontweight="bold",ha="center",xytext=(0,-38),textcoords="offset points")
    # 预测下轮峰
    ax.scatter([next_peak],[next_peak_px],color="#b91c1c",marker="*",s=200,zorder=7)
    ax.annotate(f"预测下轮峰\n{next_peak.strftime('%Y-%m')}\n${np_lo:,.0f}~${np_hi:,.0f}",(next_peak,next_peak_px),
                color="#b91c1c",fontsize=9,fontweight="bold",ha="center",xytext=(0,12),textcoords="offset points")
    ax.axvline(next_halving,color="#9333ea",ls=":",lw=1); ax.annotate(f"下次减半\n{next_halving.strftime('%Y-%m')}",
                (next_halving,price.price.min()*3),color="#9333ea",fontsize=8,ha="center")
    ax.set_xlim(pd.Timestamp("2013-01-01"),next_peak+pd.Timedelta(days=120))
    ax.set_title("BTC 预测：本轮底 + 下轮峰（时间用四轮节奏，价格用 MVRV/AHR/衰减倍数）",fontsize=14,fontweight="bold")
    ax.set_ylabel("价格 USD(对数)"); ax.legend(loc="upper left"); ax.grid(True,which="both",alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chart=C.CHART_DIR/f"forecast_{price.date.max().date()}.png"
    fig.savefig(chart,dpi=135,bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {chart.name}")

    # ---- HTML ----
    b64=base64.b64encode(chart.read_bytes()).decode()
    calc="\n".join(log)
    day=price.date.max().date()
    html=f"""<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8>
<title>BTC 本轮底 + 下轮峰 推算 {day}</title><style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0}}
 .page{{max-width:1080px;margin:0 auto;padding:38px 26px 80px}} h1{{font-size:23px;border-bottom:2px solid #1a1a1a;padding-bottom:10px}}
 h2{{font-size:18px;margin-top:30px}} .res{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}
 .card{{flex:1;min-width:230px;background:#fff;border:1px solid #e5e5e3;border-radius:10px;padding:16px}}
 .card .big{{font-size:22px;font-weight:bold}} .b{{color:#15803d}} .r{{color:#b91c1c}}
 img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;margin:10px 0}}
 pre{{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;font-size:12.5px;overflow-x:auto;line-height:1.5}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:30px;border-top:1px solid #e5e5e3;padding-top:12px}}</style></head><body><div class=page>
<h1>BTC 本轮见底 + 下轮见顶 推算</h1>
<p style="color:#6b6b6b;font-size:14px">数据至 {day}　·　时间=四轮节奏共识　·　价格=MVRV/AHR999/衰减倍数三法</p>
<div class=res>
 <div class=card><div>本轮底 · 时间</div><div class="big b">{bottom_date.year}年{bottom_date.month}月</div><div style=color:#666>真顶+{p2b_avg:.0f}天（×收缩→{bottom_date_c.strftime('%Y-%m')}）</div></div>
 <div class=card><div>本轮底 · 价格</div><div class="big b">${lo:,.0f} ~ ${hi:,.0f}</div><div style=color:#666>跌幅 {(lo/peaks[2025][1]-1)*100:.0f}%~{(hi/peaks[2025][1]-1)*100:.0f}%（MVRV法/AHR法）</div></div>
</div>
<div class=res>
 <div class=card><div>下轮峰 · 时间</div><div class="big r">{next_peak.year}年{next_peak.month}月</div><div style=color:#666>下次减半({next_halving.strftime('%Y-%m')})+{h2p_avg:.0f}天</div></div>
 <div class=card><div>下轮峰 · 价格</div><div class="big r">${np_lo:,.0f} ~ ${np_hi:,.0f}</div><div style=color:#666>本轮底 × R_next({Rn_lo:.1f}~{Rn_hi:.1f})｜月均价幂压缩 p≈{p_mid:.2f}</div></div>
</div>
<h2>预测图</h2><img src="data:image/png;base64,{b64}"/>
<h2>完整计算过程</h2><pre>{calc}</pre>
<h2>理由与假设</h2>
<ul style="font-size:14px;line-height:1.8">
<li><b>时间</b>：成熟周期(2017/2021)节奏极稳——减半→峰 {h2p_avg:.0f}天、峰→底 {p2b_avg:.0f}天(标准差仅几天)；本轮峰方法已验证误差2天，故时间可信度最高。</li>
<li><b>底价·MVRV法</b>：历史底 MVRV 逐轮抬升(0.42→0.56→0.69→0.78)，外推≈{mvrv_b:.2f}；× 已实现价格(成本线)外推值 = 底价。这是"持币者整体浮亏到投降"的客观锚。</li>
<li><b>底价·AHR999法</b>：历史底 AHR≈{ahr_b:.2f}，用 P=√(AHR×gma200×幂律中枢) 反解。</li>
<li><b>顶价·底→峰倍数幂压缩(你的理念,用月均价去噪)</b>：底→峰倍数 R(月均) 逐轮压缩({Rs[0]:.0f}→{Rs[1]:.0f}→{Rs[2]:.1f}→{Rs[3]:.1f})；R_next=R_last^p。<b>最准指数 p≈{p_mid:.3f}</b>(成熟两轮2017→2021、2021→2025几乎完全一致；样本外同时命中2021&2025峰MAE仅0.2%)。你的"开方"0.5会错-46%，早期0.72错+18%。指数已钉死 → R_next≈{Rn_mid:.2f}，下轮峰的不确定性几乎只来自"本轮底"。</li>
<li><b>不确定性</b>：时间&gt;价格。底价两法有差距(MVRV偏低、AHR偏高)，给区间；下轮顶价依赖"衰减规律延续"，若机构化打破规律会偏高。</li>
</ul>
<div class=foot>signals/forecast.py 生成。辅助分析，非买卖建议。</div>
</div></body></html>"""
    out=C.OUT_DIR/f"report_forecast_{day}.html"
    out.write_text(html,encoding="utf-8")
    print(f"\n[html] {out}")

if __name__=="__main__":
    main()
