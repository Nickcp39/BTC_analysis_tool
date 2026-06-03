from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
TABLE_DIR = OUT_DIR / "tables"
PNG_DIR = OUT_DIR / "png"
DATA_FILE = ROOT / "data" / "btc_merged_daily.csv"

HALVINGS = {
    "2012": pd.Timestamp("2012-11-28"),
    "2016": pd.Timestamp("2016-07-09"),
    "2020": pd.Timestamp("2020-05-11"),
    "2024": pd.Timestamp("2024-04-20"),
}
PEAKS = {
    "2017": pd.Timestamp("2017-12-17"),
    "2021": pd.Timestamp("2021-11-10"),
}
KNOWN_BOTTOMS = {
    "2015": pd.Timestamp("2015-01-14"),
    "2018": pd.Timestamp("2018-12-15"),
    "2022": pd.Timestamp("2022-11-21"),
}

# Original project memory: volatility levels decline by cycle.
# The user's correction here is the key: apply sqrt in log space.
VOL_LEVEL = {"2017": 9.0, "2021": 3.0, "2025": 1.0}
VOL_ALPHA = 0.5


@dataclass
class Component:
    name: str
    bottom_date: pd.Timestamp
    bottom_price: float
    sigma_days: float
    sigma_price_pct: float
    weight: float
    note: str


def load_series() -> pd.Series:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df.set_index("date")["price"].astype(float).sort_index()


def price_at_or_before(s: pd.Series, date: pd.Timestamp) -> float:
    return float(s.loc[:date].iloc[-1])


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def weighted_quantile(x: np.ndarray, w: np.ndarray, qs: list[float]) -> list[float]:
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return [float(np.interp(q, cdf, x)) for q in qs]


def sample_component(c: Component, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dates = rng.normal(c.bottom_date.toordinal(), c.sigma_days, n)
    prices = rng.lognormal(np.log(c.bottom_price), np.log1p(c.sigma_price_pct), n)
    return dates, prices


def component_frame(components: list[Component]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": c.name,
                "bottom_date": c.bottom_date.date().isoformat(),
                "bottom_price": round(c.bottom_price, 2),
                "sigma_days": c.sigma_days,
                "sigma_price_pct": c.sigma_price_pct,
                "weight": c.weight,
                "note": c.note,
            }
            for c in components
        ]
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    s = load_series()
    latest_date = s.index.max().normalize()
    latest_price = float(s.loc[latest_date])

    post_2024 = s.loc[HALVINGS["2024"] : latest_date]
    peak25_date = post_2024.idxmax().normalize()
    peak25_price = float(post_2024.max())
    current_day = int((latest_date - peak25_date).days)
    current_log_dd = float(np.log(latest_price / peak25_price))

    hist_rows = []
    for cycle, peak_date in PEAKS.items():
        next_halving = HALVINGS["2020"] if cycle == "2017" else HALVINGS["2024"]
        window = s.loc[peak_date : next_halving - pd.Timedelta(days=1)]
        bottom_date = window.idxmin().normalize()
        peak_price = price_at_or_before(s, peak_date)
        bottom_price = float(window.min())
        raw_log_dd = float(np.log(bottom_price / peak_price))
        sqrt_scale = (VOL_LEVEL["2025"] / VOL_LEVEL[cycle]) ** VOL_ALPHA
        annealed_log_dd = raw_log_dd * sqrt_scale
        annealed_price = peak25_price * np.exp(annealed_log_dd)
        hist_rows.append(
            {
                "cycle": cycle,
                "peak_date": peak_date.date().isoformat(),
                "peak_price": peak_price,
                "bottom_date": bottom_date.date().isoformat(),
                "bottom_price": bottom_price,
                "days_peak_to_bottom": int((bottom_date - peak_date).days),
                "raw_bottom_ratio": bottom_price / peak_price,
                "raw_log_drawdown": raw_log_dd,
                "sqrt_vol_scale_to_2025": sqrt_scale,
                "annealed_log_drawdown": annealed_log_dd,
                "annealed_2025_bottom_price": annealed_price,
            }
        )
    hist = pd.DataFrame(hist_rows)
    hist.to_csv(TABLE_DIR / "sqrt_log_historical_components.csv", index=False)

    bottom_to_next_halving = np.array(
        [
            (HALVINGS["2016"] - KNOWN_BOTTOMS["2015"]).days,
            (HALVINGS["2020"] - KNOWN_BOTTOMS["2018"]).days,
            (HALVINGS["2024"] - KNOWN_BOTTOMS["2022"]).days,
        ],
        dtype=float,
    )
    halving_intervals = np.array(
        [
            (HALVINGS["2016"] - HALVINGS["2012"]).days,
            (HALVINGS["2020"] - HALVINGS["2016"]).days,
            (HALVINGS["2024"] - HALVINGS["2020"]).days,
        ],
        dtype=float,
    )
    next_halving_interval = float(np.polyval(np.polyfit([1, 2, 3], halving_intervals, 1), 4))
    next_halving_reg = HALVINGS["2024"] + pd.Timedelta(days=round(next_halving_interval))
    next_halving_flat = HALVINGS["2024"] + pd.Timedelta(days=1440)
    b2h_mean = float(bottom_to_next_halving.mean())
    b2h_std = float(bottom_to_next_halving.std(ddof=0))

    days = hist["days_peak_to_bottom"].to_numpy(dtype=float)
    sqrt_log_prices = hist["annealed_2025_bottom_price"].to_numpy(dtype=float)
    sqrt_log_center_price = float(np.mean(sqrt_log_prices))

    # A light calibration: current observed drawdown already exceeded the 2017-sqrt replay,
    # so that component receives wider uncertainty rather than forcing the center upward.
    components = [
        Component(
            name="sqrt_log_replay_2017",
            bottom_date=peak25_date + pd.Timedelta(days=int(hist.loc[hist["cycle"] == "2017", "days_peak_to_bottom"].iloc[0])),
            bottom_price=float(hist.loc[hist["cycle"] == "2017", "annealed_2025_bottom_price"].iloc[0]),
            sigma_days=24.0,
            sigma_price_pct=0.16,
            weight=0.70,
            note="log(bottom/peak) from 2017 multiplied by sqrt(1/9), then replayed from 2025 peak",
        ),
        Component(
            name="sqrt_log_replay_2021",
            bottom_date=peak25_date + pd.Timedelta(days=int(hist.loc[hist["cycle"] == "2021", "days_peak_to_bottom"].iloc[0])),
            bottom_price=float(hist.loc[hist["cycle"] == "2021", "annealed_2025_bottom_price"].iloc[0]),
            sigma_days=24.0,
            sigma_price_pct=0.14,
            weight=1.25,
            note="log(bottom/peak) from 2021 multiplied by sqrt(1/3), then replayed from 2025 peak",
        ),
        Component(
            name="sqrt_log_blended_time",
            bottom_date=peak25_date + pd.Timedelta(days=round(days.mean())),
            bottom_price=sqrt_log_center_price,
            sigma_days=max(20.0, float(days.std(ddof=0)) + 18.0),
            sigma_price_pct=0.13,
            weight=1.00,
            note="mean of sqrt-log replay prices with mean 2017/2021 peak-to-bottom time",
        ),
        Component(
            name="bottom_to_next_halving_conservative",
            bottom_date=next_halving_flat - pd.Timedelta(days=round(b2h_mean)),
            bottom_price=sqrt_log_center_price * 0.96,
            sigma_days=max(30.0, b2h_std + 24.0),
            sigma_price_pct=0.15,
            weight=0.80,
            note="stable bottom-to-next-halving interval, using sqrt-log price anchor",
        ),
        Component(
            name="bottom_to_next_halving_regressed",
            bottom_date=next_halving_reg - pd.Timedelta(days=round(b2h_mean)),
            bottom_price=sqrt_log_center_price * 0.92,
            sigma_days=max(44.0, b2h_std + 32.0),
            sigma_price_pct=0.17,
            weight=0.45,
            note="slower next halving regression; keeps late-date tail but lower weight",
        ),
    ]
    comp_df = component_frame(components)
    comp_df.to_csv(TABLE_DIR / "model_components.csv", index=False)

    rng = np.random.default_rng(20260601)
    date_samples, price_samples, weight_samples = [], [], []
    for c in components:
        d, p = sample_component(c, 50000, rng)
        date_samples.append(d)
        price_samples.append(p)
        weight_samples.append(np.full_like(d, c.weight, dtype=float))
    date_samples = np.concatenate(date_samples)
    price_samples = np.concatenate(price_samples)
    weights = np.concatenate(weight_samples)

    date_q = weighted_quantile(date_samples, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    price_q = weighted_quantile(price_samples, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    center_date = pd.Timestamp.fromordinal(round(weighted_mean(date_samples, weights)))
    center_price = weighted_mean(price_samples, weights)

    monthly = pd.DataFrame(
        {
            "date": [pd.Timestamp.fromordinal(round(x)) for x in date_samples],
            "weight": weights,
        }
    )
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_prob = monthly.groupby("month")["weight"].sum().reset_index()
    monthly_prob["probability"] = monthly_prob["weight"] / monthly_prob["weight"].sum()
    monthly_prob.drop(columns=["weight"]).to_csv(TABLE_DIR / "monthly_bottom_probability.csv", index=False)

    plt.figure(figsize=(11, 5.8))
    plt.hist([pd.Timestamp.fromordinal(round(x)) for x in date_samples], bins=42, weights=weights, color="#4c78a8", alpha=0.78)
    plt.axvline(center_date, color="#111111", linewidth=2, label=f"center {center_date.date()}")
    plt.axvspan(pd.Timestamp.fromordinal(round(date_q[0])), pd.Timestamp.fromordinal(round(date_q[-1])), color="#f58518", alpha=0.20, label="10-90%")
    plt.title("Sqrt-log bottom date distribution")
    plt.xlabel("date")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_date_distribution.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    plt.hist(price_samples, bins=55, weights=weights, color="#54a24b", alpha=0.78)
    plt.axvline(center_price, color="#111111", linewidth=2, label=f"center ${center_price:,.0f}")
    plt.axvspan(price_q[0], price_q[-1], color="#f58518", alpha=0.20, label="10-90%")
    plt.axvline(latest_price, color="#1a5276", linestyle="--", label=f"latest ${latest_price:,.0f}")
    plt.title("Sqrt-log bottom price distribution")
    plt.xlabel("BTC price")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_price_distribution.png", dpi=170)
    plt.close()

    # Draw log drawdown replay against observed path.
    plt.figure(figsize=(12, 6.5))
    obs_idx = pd.date_range(peak25_date, latest_date, freq="D")
    obs = s.reindex(obs_idx).ffill()
    obs_log = np.log(obs / peak25_price)
    plt.plot((obs.index - peak25_date).days, obs_log, label="2025 observed log drawdown", color="#1a5276", linewidth=2.6)
    for _, row in hist.iterrows():
        x = [0, row["days_peak_to_bottom"]]
        y = [0, row["annealed_log_drawdown"]]
        plt.plot(x, y, linestyle="--", linewidth=2, label=f"{row['cycle']} sqrt-log replay")
        plt.scatter([row["days_peak_to_bottom"]], [row["annealed_log_drawdown"]], s=45)
    plt.axvline(current_day, color="#1a5276", linestyle=":", alpha=0.85)
    plt.title("Observed vs sqrt-log annealed drawdown targets")
    plt.xlabel("days after 2025 actual peak")
    plt.ylabel("log(price / peak)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "sqrt_log_drawdown_replay.png", dpi=170)
    plt.close()

    summary = {
        "generated": "2026-06-01",
        "model": "bottom_model_sqrt_log_v1",
        "data_range": [s.index.min().date().isoformat(), latest_date.date().isoformat()],
        "latest": {"date": latest_date.date().isoformat(), "price": latest_price},
        "actual_2025_peak": {
            "date": peak25_date.date().isoformat(),
            "price": peak25_price,
            "current_day_after_peak": current_day,
            "current_log_drawdown": current_log_dd,
            "current_drawdown_pct": (latest_price / peak25_price - 1.0) * 100.0,
        },
        "formula": "annealed_log_drawdown = log(historical_bottom / historical_peak) * sqrt(vol_2025 / vol_history)",
        "vol_levels": VOL_LEVEL,
        "bottom_forecast": {
            "center_date": center_date.date().isoformat(),
            "date_p10": pd.Timestamp.fromordinal(round(date_q[0])).date().isoformat(),
            "date_p25": pd.Timestamp.fromordinal(round(date_q[1])).date().isoformat(),
            "date_p50": pd.Timestamp.fromordinal(round(date_q[2])).date().isoformat(),
            "date_p75": pd.Timestamp.fromordinal(round(date_q[3])).date().isoformat(),
            "date_p90": pd.Timestamp.fromordinal(round(date_q[4])).date().isoformat(),
            "center_price": center_price,
            "price_p10": price_q[0],
            "price_p25": price_q[1],
            "price_p50": price_q[2],
            "price_p75": price_q[3],
            "price_p90": price_q[4],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# BTC Bottom Model Sqrt-Log V1",
        "",
        "Generated: 2026-06-01",
        f"Data: {s.index.min().date()} to {latest_date.date()}",
        f"Latest BTC: ${latest_price:,.2f} on {latest_date.date()}",
        "",
        "## Core formula",
        "`annealed_log_drawdown = log(historical_bottom / historical_peak) * sqrt(vol_2025 / vol_history)`",
        "",
        "This is the user's corrected core: volatility decays by cycle, and the historical amplitude is compressed in log space with a square-root factor.",
        "",
        "## Peak anchor",
        f"- Actual 2025 cycle high since 2024 halving: {peak25_date.date()} at ${peak25_price:,.2f}",
        f"- Current position: day {current_day} after peak, drawdown {(latest_price / peak25_price - 1.0) * 100.0:.2f}%, log drawdown {current_log_dd:.4f}",
        "",
        "## New bottom forecast",
        f"- Center date: {center_date.date()}",
        f"- Date band 10-90%: {pd.Timestamp.fromordinal(round(date_q[0])).date()} to {pd.Timestamp.fromordinal(round(date_q[-1])).date()}",
        f"- Date band 25-75%: {pd.Timestamp.fromordinal(round(date_q[1])).date()} to {pd.Timestamp.fromordinal(round(date_q[3])).date()}",
        f"- Center price: ${center_price:,.0f}",
        f"- Price band 10-90%: ${price_q[0]:,.0f} to ${price_q[-1]:,.0f}",
        f"- Price band 25-75%: ${price_q[1]:,.0f} to ${price_q[3]:,.0f}",
        "",
        "## Historical sqrt-log components",
        hist.to_string(index=False),
        "",
        "## Model components",
        comp_df.to_string(index=False),
    ]
    (OUT_DIR / "bottom_model_sqrt_log_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
