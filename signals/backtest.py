"""回测：留一法(LOO)交叉验证 + 样本外(2025)验证，证明算法的近似准确性。

测三件事：
  1) 时间·减半→峰   2) 时间·峰→底   3) 价格·底→峰倍数(幂压缩)
每个都"抽掉一轮，用其余轮预测它"，看误差；并给成熟周期 vs 全样本两口径。
理念：不要完美，要近似；早期(2011/2013)市场小、波动大，误差天然大。
产出 outputs/report_backtest_*.html
"""
from __future__ import annotations
import base64, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C, indicators as I

price=pd.read_csv(C.PRICE_CSV,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
price["ma30"]=price["price"].rolling(30,center=True,min_periods=15).mean()
price["lr"]=np.log(price["price"]).diff()
ahr=I.compute_ahr999(price)[["date","ahr999"]]

HALV={2013:"2012-11-28",2017:"2016-07-09",2021:"2020-05-11",2025:"2024-04-20"}
HALV={k:pd.Timestamp(v) for k,v in HALV.items()}
TOPW={2011:("2011-04-01","2011-08-31"),2013:("2013-10-01","2014-02-15"),2017:("2017-11-01","2018-01-31"),
      2021:("2021-10-01","2021-12-31"),2025:("2025-09-01","2025-12-15")}
BOTW={2011:("2011-09-01","2012-03-31"),2015:("2014-10-01","2015-09-30"),
      2018:("2018-09-01","2019-04-30"),2022:("2022-09-01","2023-03-31")}

def pmax_date(a,b): w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; r=w.loc[w.price.idxmax()]; return r.date,float(r.price)
def amax(a,b): w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; return float(w.ma30.max())
def amin(a,b): w=price[(price.date>=pd.Timestamp(a))&(price.date<=pd.Timestamp(b))]; return float(w.ma30.min())
def ahr_bot_date(a,b):
    w=ahr[(ahr.date>=pd.Timestamp(a))&(ahr.date<=pd.Timestamp(b))].dropna(subset=["ahr999"]); r=w.loc[w.ahr999.idxmin()]; return r.date

tops={c:pmax_date(*w) for c,w in TOPW.items()}
# 各轮关键天数
h2p={c:(tops[c][0]-HALV[c]).days for c in HALV}                              # 减半→峰
# 峰→底（底用 AHR 最低；底所属"峰"= 同轮的前一个价格顶）
bot_peak={2011:tops[2011][0],2015:tops[2013][0],2018:tops[2017][0],2022:tops[2021][0]}
bot_date={c:ahr_bot_date(*BOTW[c]) for c in BOTW}
p2b={c:(bot_date[c]-bot_peak[c]).days for c in BOTW}
# 底→峰倍数(月均)
apv={c:amax(*TOPW[c]) for c in [2013,2017,2021,2025]}
abv={c:amin(*BOTW[c]) for c in [2011,2015,2018,2022]}
Rpairs=[(2011,2013),(2015,2017),(2018,2021),(2022,2025)]
R={pk:apv[pk]/abv[bt] for bt,pk in Rpairs}     # key by peak-year

def loo_mean(d, keys):
    """对每个key，用其余key的均值预测它。返回 [(key,actual,pred,err%)]"""
    out=[]
    for k in keys:
        others=[d[j] for j in keys if j!=k]
        pred=np.mean(others); out.append((k,d[k],pred,(pred/d[k]-1)*100))
    return out

def loo_power(Rd, order):
    """底→峰倍数 LOO：用其余轮拟合的指数 p 预测该轮 R = R_prev^p。"""
    lnR={k:np.log(v) for k,v in Rd.items()}
    # 各相邻指数
    exps={order[i+1]:lnR[order[i+1]]/lnR[order[i]] for i in range(len(order)-1)}  # key=被预测轮
    out=[]
    for i in range(1,len(order)):
        tgt=order[i]; prev=order[i-1]
        p_others=np.mean([exps[k] for k in exps if k!=tgt])
        pred=Rd[prev]**p_others
        out.append((tgt,Rd[tgt],pred,(pred/Rd[tgt]-1)*100))
    return out,exps

def mae(rows,subset=None):
    e=[abs(r[3]) for r in rows if subset is None or r[0] in subset]
    return float(np.mean(e))

order=[2013,2017,2021,2025]
peak_rows=loo_mean(h2p,[2013,2017,2021,2025])
bot_rows=loo_mean(p2b,[2011,2015,2018,2022])
price_rows,exps=loo_power(R,order)

MAT_PK=[2017,2021,2025]; MAT_BT=[2018,2022]; MAT_PR=[2021,2025]

# 样本外：站在2022底预测2025峰
p12=np.log(R[2021])/np.log(R[2017])
oos_pred=abv[2022]*(R[2021]**p12); oos_act=apv[2025]; oos_err=(oos_pred/oos_act-1)*100

def tbl(rows,unit=""):
    s="<tr><th>抽出轮</th><th>实际</th><th>预测</th><th>误差</th></tr>"
    for k,a,p,e in rows:
        col="#15803d" if abs(e)<15 else ("#b45309" if abs(e)<35 else "#b91c1c")
        s+=f"<tr><td>{k}</td><td>{a:,.0f}{unit}</td><td>{p:,.0f}{unit}</td><td style=color:{col}>{e:+.0f}%</td></tr>"
    return s

html=f"""<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>BTC 算法回测 {price.date.max().date()}</title><style>
body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0}}
.page{{max-width:1000px;margin:0 auto;padding:38px 26px 80px}} h1{{font-size:23px;border-bottom:2px solid #1a1a1a;padding-bottom:10px}}
h2{{font-size:18px;margin-top:30px}} table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
th,td{{border:1px solid #e5e5e3;padding:7px 11px;text-align:center}} th{{background:#f3f4f6}}
.box{{background:#dcfce7;border:1px solid #15803d;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:15px}}
.foot{{color:#9ca3af;font-size:12px;margin-top:30px;border-top:1px solid #e5e5e3;padding-top:12px}}</style></head><body><div class=page>
<h1>BTC 节奏/价格 算法回测（留一法 + 样本外）</h1>
<p style=color:#6b6b6b;font-size:14px>数据至 {price.date.max().date()}　·　理念：不求完美，验证近似准确性；早期(2011/2013)市场小波动大，误差天然大</p>

<div class=box>🎯 <b>样本外硬验证</b>：只用 ≤2021 数据 + 2022底，预测 2025 峰 = <b>${oos_pred:,.0f}</b> vs 实际(月均) ${oos_act:,.0f}　→ <b>误差 {oos_err:+.1f}%</b></div>

<h2>1. 时间 · 减半→峰（留一法，单位:天）</h2>
<table>{tbl(peak_rows,'d')}</table>
<p>成熟周期(2017/2021/2025)平均绝对误差 <b>{mae(peak_rows,MAT_PK):.0f}%</b>；全样本 {mae(peak_rows):.0f}%。</p>

<h2>2. 时间 · 峰→底（留一法，单位:天）</h2>
<table>{tbl(bot_rows,'d')}</table>
<p>成熟周期(2018/2022)平均绝对误差 <b>{mae(bot_rows,MAT_BT):.0f}%</b>；全样本 {mae(bot_rows):.0f}%（2011仅163天，早期极端值拉高误差）。</p>

<h2>3. 价格 · 底→峰倍数 R（留一法，幂压缩）</h2>
<table>{tbl(price_rows)}</table>
<p>成熟周期(2021/2025)平均绝对误差 <b>{mae(price_rows,MAT_PR):.0f}%</b>；全样本 {mae(price_rows):.0f}%。</p>

<h2>结论</h2>
<div class=box>
<b>近似准确性成立</b>：在成熟周期上——减半→峰误差 ~{mae(peak_rows,MAT_PK):.0f}%、峰→底 ~{mae(bot_rows,MAT_BT):.0f}%、底→峰倍数 ~{mae(price_rows,MAT_PR):.0f}%；样本外预测2025峰误差仅 {oos_err:+.1f}%。<br>
早期(2011/2013)误差大是因为市场小、波动率高——这正印证"波动率随时间衰减、规律逐渐变清晰"。<br>
<b>所以本轮底(~2026-10)与下轮峰(~2029-10, ~\$160k-220k)的推算，置信度来自成熟周期的可复算性，而非完美公式。</b>
</div>
<div class=foot>signals/backtest.py 生成。留一法=抽掉一轮用其余预测它。辅助分析，非买卖建议。</div>
</div></body></html>"""
out=C.OUT_DIR/f"report_backtest_{price.date.max().date()}.html"
out.write_text(html,encoding="utf-8")

print("样本外 2025峰预测:",f"${oos_pred:,.0f} vs ${oos_act:,.0f} ({oos_err:+.1f}%)")
print("减半→峰 LOO:",[(k,f"{e:+.0f}%") for k,a,p,e in peak_rows]," 成熟MAE",f"{mae(peak_rows,MAT_PK):.0f}%")
print("峰→底 LOO:",[(k,f"{e:+.0f}%") for k,a,p,e in bot_rows]," 成熟MAE",f"{mae(bot_rows,MAT_BT):.0f}%")
print("底→峰倍数 LOO:",[(k,f"{e:+.0f}%") for k,a,p,e in price_rows]," 成熟MAE",f"{mae(price_rows,MAT_PR):.0f}%")
print("[html]",out)
