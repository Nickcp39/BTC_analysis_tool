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
KNOWN_BOTTOMS = {
    "2015": pd.Timestamp("2015-01-14"),
    "2018": pd.Timestamp("2018-12-15"),
    "2022": pd.Timestamp("2022-11-21"),
}
KNOWN_PEAKS = {
    "2017": pd.Timestamp("2017-12-17"),
    "2021": pd.Timestamp("2021-11-10"),
}

VOL_LEVEL = {"2017": 9.0, "2021": 3.0, "2025": 1.0}
VOL_ALPHA = 0.5


@dataclass
class Observation:
    cycle: str
    peak_date: pd.Timestamp
    peak_price: float
    bottom_date: pd.Timestamp
    bottom_price: float

    @property
    def days_peak_to_bottom(self) -> int:
        return int((self.bottom_date - self.peak_date).days)

    @property
    def bottom_to_peak_ratio(self) -> float:
        return self.bottom_price / self.peak_price

    @property
    def log_bottom_to_peak_ratio(self) -> float:
        return float(np.log(self.bottom_to_peak_ratio))


@dataclass
class PriceModel:
    name: str
    bottom_price: float
    sigma_pct: float
    weight: float
    note: str


@dataclass
class TimeModel:
    name: str
    bottom_date: pd.Timestamp
    sigma_days: float
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


def sample_price(model: PriceModel, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.lognormal(np.log(model.bottom_price), np.log1p(model.sigma_pct), n)


def sample_date(model: TimeModel, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(model.bottom_date.toordinal(), model.sigma_days, n)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    s = load_series()
    latest_date = s.index.max().normalize()
    latest_price = float(s.loc[latest_date])

    post_2024 = s.loc[HALVINGS["2024"] : latest_date]
    peak_2025_date = post_2024.idxmax().normalize()
    peak_2025_price = float(post_2024.max())
    current_day = int((latest_date - peak_2025_date).days)
    current_dd = latest_price / peak_2025_price - 1.0

    observations = []
    for cycle in ("2017", "2021"):
        peak_date = KNOWN_PEAKS[cycle]
        next_halving = HALVINGS["2020"] if cycle == "2017" else HALVINGS["2024"]
        window = s.loc[peak_date : next_halving - pd.Timedelta(days=1)]
        bottom_date = window.idxmin().normalize()
        observations.append(
            Observation(
                cycle=cycle,
                peak_date=peak_date,
                peak_price=price_at_or_before(s, peak_date),
                bottom_date=bottom_date,
                bottom_price=float(window.min()),
            )
        )

    obs_df = pd.DataFrame(
        [
            {
                "cycle": o.cycle,
                "peak_date": o.peak_date.date().isoformat(),
                "peak_price": o.peak_price,
                "bottom_date": o.bottom_date.date().isoformat(),
                "bottom_price": o.bottom_price,
                "days_peak_to_bottom": o.days_peak_to_bottom,
                "bottom_to_peak_ratio": o.bottom_to_peak_ratio,
                "log_bottom_to_peak_ratio": o.log_bottom_to_peak_ratio,
            }
            for o in observations
        ]
    )
    obs_df.to_csv(TABLE_DIR / "historical_bottom_ratio_observations.csv", index=False)

    cycles = np.array([1.0, 2.0])
    ratio = np.array([o.bottom_to_peak_ratio for o in observations])
    log_ratio = np.log(ratio)
    bottom_prices = np.array([o.bottom_price for o in observations])
    days_peak_to_bottom = np.array([o.days_peak_to_bottom for o in observations], dtype=float)

    # 1) Bottom/peak ratio model, analogous to the old peak multiplier model.
    #    Use log ratio because the old core model worked in log space for multiplicative quantities.
    a_lr, b_lr = np.polyfit(cycles, log_ratio, 1)
    pred_log_ratio_trend = float(a_lr * 3.0 + b_lr)
    pred_ratio_trend = float(np.exp(pred_log_ratio_trend))
    price_ratio_log_trend = peak_2025_price * pred_ratio_trend

    # 2) Bottom-to-bottom log trend, analogous to top-to-top and bottom-to-bottom soft constraints.
    a_b, b_b = np.polyfit(np.array([1.0, 2.0]), np.log(bottom_prices), 1)
    price_bottom_log_trend = float(np.exp(a_b * 3.0 + b_b))

    # 3) Sqrt volatility annealing in log drawdown space.
    sqrt_replay_rows = []
    for o in observations:
        scale = (VOL_LEVEL["2025"] / VOL_LEVEL[o.cycle]) ** VOL_ALPHA
        ann_log_ratio = o.log_bottom_to_peak_ratio * scale
        sqrt_replay_rows.append(
            {
                "cycle": o.cycle,
                "raw_log_ratio": o.log_bottom_to_peak_ratio,
                "sqrt_vol_scale": scale,
                "annealed_log_ratio": ann_log_ratio,
                "annealed_ratio": float(np.exp(ann_log_ratio)),
                "annealed_price_from_2025_peak": float(peak_2025_price * np.exp(ann_log_ratio)),
                "bottom_date_by_replay": (peak_2025_date + pd.Timedelta(days=o.days_peak_to_bottom)).date().isoformat(),
            }
        )
    sqrt_df = pd.DataFrame(sqrt_replay_rows)
    sqrt_df.to_csv(TABLE_DIR / "sqrt_log_ratio_replay.csv", index=False)
    price_sqrt_log_mean = float(sqrt_df["annealed_price_from_2025_peak"].mean())
    price_sqrt_log_2021 = float(sqrt_df.loc[sqrt_df["cycle"] == "2021", "annealed_price_from_2025_peak"].iloc[0])

    # 4) Old core bottom-price ensemble, but used as a soft anchor, not a hard override.
    old_core_prices = np.array([48369.0, 50894.0, 48770.0])
    old_core_weights = np.array([0.9, 1.0, 0.9])
    price_old_core = weighted_mean(old_core_prices, old_core_weights)

    price_models = [
        PriceModel(
            "bottom_peak_log_ratio_trend",
            price_ratio_log_trend,
            0.14,
            1.15,
            "log(bottom/peak ratio) trend from 2017->2021 extrapolated one cycle",
        ),
        PriceModel(
            "bottom_to_bottom_log_trend",
            price_bottom_log_trend,
            0.18,
            0.85,
            "log(bottom price) trend from 2018->2022 extrapolated to next bottom",
        ),
        PriceModel(
            "sqrt_log_vol_annealed_mean",
            price_sqrt_log_mean,
            0.15,
            1.20,
            "historical log drawdown compressed by sqrt volatility scale, mean of 2017/2021",
        ),
        PriceModel(
            "sqrt_log_vol_annealed_2021",
            price_sqrt_log_2021,
            0.13,
            1.10,
            "2021-only sqrt-log replay; best shape match in prior diagnostics",
        ),
        PriceModel(
            "old_core_bottom_soft_anchor",
            price_old_core,
            0.16,
            0.75,
            "prior core artifact bottom ensemble, retained as soft anchor",
        ),
    ]
    price_df = pd.DataFrame([m.__dict__ for m in price_models])
    price_df.to_csv(TABLE_DIR / "price_models.csv", index=False)

    # Time models: mirror old top timing, but for bottom.
    b2h = np.array(
        [
            (HALVINGS["2016"] - KNOWN_BOTTOMS["2015"]).days,
            (HALVINGS["2020"] - KNOWN_BOTTOMS["2018"]).days,
            (HALVINGS["2024"] - KNOWN_BOTTOMS["2022"]).days,
        ],
        dtype=float,
    )
    h2h = np.array(
        [
            (HALVINGS["2016"] - HALVINGS["2012"]).days,
            (HALVINGS["2020"] - HALVINGS["2016"]).days,
            (HALVINGS["2024"] - HALVINGS["2020"]).days,
        ],
        dtype=float,
    )
    next_halving_interval_reg = float(np.polyval(np.polyfit([1, 2, 3], h2h, 1), 4))
    next_halving_reg = HALVINGS["2024"] + pd.Timedelta(days=round(next_halving_interval_reg))
    next_halving_flat = HALVINGS["2024"] + pd.Timedelta(days=1440)
    b2h_mean = float(b2h.mean())
    b2h_std = float(b2h.std(ddof=0))
    p2b_mean = float(days_peak_to_bottom.mean())
    p2b_std = float(days_peak_to_bottom.std(ddof=0))

    # For two points, regression is equivalent to extrapolating the slow extension from 363->376.
    p2b_reg = float(np.polyval(np.polyfit([1, 2], days_peak_to_bottom, 1), 3))

    time_models = [
        TimeModel(
            "peak_to_bottom_mean",
            peak_2025_date + pd.Timedelta(days=round(p2b_mean)),
            max(18.0, p2b_std + 14.0),
            1.05,
            "mean of 2017/2021 peak-to-bottom days",
        ),
        TimeModel(
            "peak_to_bottom_regression",
            peak_2025_date + pd.Timedelta(days=round(p2b_reg)),
            30.0,
            0.85,
            "linear extension of peak-to-bottom days, analogous to top timing regression",
        ),
        TimeModel(
            "bottom_to_next_halving_stable",
            next_halving_flat - pd.Timedelta(days=round(b2h_mean)),
            max(28.0, b2h_std + 20.0),
            1.00,
            "bottom->next halving has been stable near 524 days",
        ),
        TimeModel(
            "bottom_to_next_halving_regressed",
            next_halving_reg - pd.Timedelta(days=round(b2h_mean)),
            max(44.0, b2h_std + 32.0),
            0.45,
            "late tail if next halving interval keeps lengthening",
        ),
    ]
    time_df = pd.DataFrame(
        [
            {
                "name": m.name,
                "bottom_date": m.bottom_date.date().isoformat(),
                "sigma_days": m.sigma_days,
                "weight": m.weight,
                "note": m.note,
            }
            for m in time_models
        ]
    )
    time_df.to_csv(TABLE_DIR / "time_models.csv", index=False)

    rng = np.random.default_rng(20260601)
    price_samples, price_weights = [], []
    for m in price_models:
        p = sample_price(m, 60000, rng)
        price_samples.append(p)
        price_weights.append(np.full_like(p, m.weight, dtype=float))
    price_samples = np.concatenate(price_samples)
    price_weights = np.concatenate(price_weights)

    date_samples, date_weights = [], []
    for m in time_models:
        d = sample_date(m, 60000, rng)
        date_samples.append(d)
        date_weights.append(np.full_like(d, m.weight, dtype=float))
    date_samples = np.concatenate(date_samples)
    date_weights = np.concatenate(date_weights)

    p_q = weighted_quantile(price_samples, price_weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    d_q = weighted_quantile(date_samples, date_weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    price_center = weighted_mean(price_samples, price_weights)
    date_center = pd.Timestamp.fromordinal(round(weighted_mean(date_samples, date_weights)))

    monthly = pd.DataFrame({"date": [pd.Timestamp.fromordinal(round(x)) for x in date_samples], "weight": date_weights})
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_prob = monthly.groupby("month")["weight"].sum().reset_index()
    monthly_prob["probability"] = monthly_prob["weight"] / monthly_prob["weight"].sum()
    monthly_prob.drop(columns=["weight"]).to_csv(TABLE_DIR / "monthly_bottom_probability.csv", index=False)

    plt.figure(figsize=(11.5, 5.8))
    plt.hist(price_samples, bins=60, weights=price_weights, color="#54a24b", alpha=0.78)
    plt.axvline(price_center, color="#111111", linewidth=2, label=f"center ${price_center:,.0f}")
    plt.axvspan(p_q[0], p_q[-1], color="#f58518", alpha=0.20, label="10-90%")
    plt.axvline(latest_price, color="#1a5276", linestyle="--", label=f"latest ${latest_price:,.0f}")
    plt.title("Bottom price fusion: ratio/log/sqrt-vol models")
    plt.xlabel("BTC price")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_price_distribution.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11.5, 5.8))
    plt.hist([pd.Timestamp.fromordinal(round(x)) for x in date_samples], bins=44, weights=date_weights, color="#4c78a8", alpha=0.78)
    plt.axvline(date_center, color="#111111", linewidth=2, label=f"center {date_center.date()}")
    plt.axvspan(pd.Timestamp.fromordinal(round(d_q[0])), pd.Timestamp.fromordinal(round(d_q[-1])), color="#f58518", alpha=0.20, label="10-90%")
    plt.title("Bottom date fusion")
    plt.xlabel("date")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_date_distribution.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 6.2))
    for _, row in sqrt_df.iterrows():
        plt.scatter([row["sqrt_vol_scale"]], [row["annealed_price_from_2025_peak"]], s=80, label=f"{row['cycle']} sqrt-log replay")
    for m in price_models:
        plt.axhline(m.bottom_price, alpha=0.35, linewidth=1.8, label=m.name)
    plt.axhspan(p_q[1], p_q[3], color="#54a24b", alpha=0.12, label="25-75% fused band")
    plt.title("Bottom price model anchors")
    plt.xlabel("sqrt volatility scale, where applicable")
    plt.ylabel("bottom price")
    plt.grid(alpha=0.22)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PNG_DIR / "price_model_anchors.png", dpi=170)
    plt.close()

    summary = {
        "generated": "2026-06-01",
        "model": "bottom_ratio_fusion_v1",
        "data_range": [s.index.min().date().isoformat(), latest_date.date().isoformat()],
        "latest": {"date": latest_date.date().isoformat(), "price": latest_price},
        "actual_2025_peak": {
            "date": peak_2025_date.date().isoformat(),
            "price": peak_2025_price,
            "current_day_after_peak": current_day,
            "current_drawdown_pct": current_dd * 100.0,
        },
        "core_logic": [
            "bottom/peak ratio is modeled in log space, analogous to prior peak multiplier ratio",
            "historical log drawdown is compressed by sqrt(vol_2025 / vol_history)",
            "bottom time fuses peak-to-bottom and bottom-to-next-halving clocks",
        ],
        "bottom_forecast": {
            "center_date": date_center.date().isoformat(),
            "date_p10": pd.Timestamp.fromordinal(round(d_q[0])).date().isoformat(),
            "date_p25": pd.Timestamp.fromordinal(round(d_q[1])).date().isoformat(),
            "date_p50": pd.Timestamp.fromordinal(round(d_q[2])).date().isoformat(),
            "date_p75": pd.Timestamp.fromordinal(round(d_q[3])).date().isoformat(),
            "date_p90": pd.Timestamp.fromordinal(round(d_q[4])).date().isoformat(),
            "center_price": price_center,
            "price_p10": p_q[0],
            "price_p25": p_q[1],
            "price_p50": p_q[2],
            "price_p75": p_q[3],
            "price_p90": p_q[4],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# BTC Bottom Ratio Fusion V1",
        "",
        "Generated: 2026-06-01",
        f"Data: {s.index.min().date()} to {latest_date.date()}",
        f"Latest BTC: ${latest_price:,.2f} on {latest_date.date()}",
        "",
        "## Why this version",
        "This is the bottom-side analogue of the old peak product: use ratio/log-ratio models for price, and independent timing clocks for date.",
        "The key correction is to apply volatility decay in log space with a square root: log drawdown * sqrt(vol_2025 / vol_history).",
        "",
        "## Current anchor",
        f"- 2025 actual peak: {peak_2025_date.date()} at ${peak_2025_price:,.2f}",
        f"- Current: day {current_day} after peak, drawdown {current_dd * 100.0:.2f}%",
        "",
        "## Bottom forecast",
        f"- Center date: {date_center.date()}",
        f"- Date band 25-75%: {pd.Timestamp.fromordinal(round(d_q[1])).date()} to {pd.Timestamp.fromordinal(round(d_q[3])).date()}",
        f"- Date band 10-90%: {pd.Timestamp.fromordinal(round(d_q[0])).date()} to {pd.Timestamp.fromordinal(round(d_q[-1])).date()}",
        f"- Center price: ${price_center:,.0f}",
        f"- Price band 25-75%: ${p_q[1]:,.0f} to ${p_q[3]:,.0f}",
        f"- Price band 10-90%: ${p_q[0]:,.0f} to ${p_q[-1]:,.0f}",
        "",
        "## Historical observations",
        obs_df.to_string(index=False),
        "",
        "## Sqrt-log replay",
        sqrt_df.to_string(index=False),
        "",
        "## Price models",
        price_df.to_string(index=False),
        "",
        "## Time models",
        time_df.to_string(index=False),
    ]
    (OUT_DIR / "bottom_ratio_fusion_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
