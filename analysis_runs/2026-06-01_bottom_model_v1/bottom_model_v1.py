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

FIXED_PEAKS = {
    "2017": pd.Timestamp("2017-12-17"),
    "2021": pd.Timestamp("2021-11-10"),
}

KNOWN_BOTTOMS = {
    "2015": pd.Timestamp("2015-01-14"),
    "2018": pd.Timestamp("2018-12-15"),
    "2022": pd.Timestamp("2022-11-21"),
}

OLD_CORE_BOTTOM_MODELS = [
    ("old_four_relation_nominal", 48369.0, 0.90),
    ("old_real_full_inflation", 50894.0, 1.00),
    ("old_combined_average", 48770.0, 0.90),
]


@dataclass
class ModelComponent:
    name: str
    bottom_date: pd.Timestamp
    bottom_price: float
    sigma_days: float
    sigma_price_pct: float
    weight: float
    note: str


def load_price_series() -> pd.Series:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    s = df.set_index("date")["price"].astype(float).sort_index()
    return s


def price_at_or_before(s: pd.Series, ts: pd.Timestamp) -> float:
    return float(s.loc[:ts].iloc[-1])


def rel_curve(s: pd.Series, peak_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    idx = pd.date_range(peak_date, end_date, freq="D")
    ss = s.reindex(idx).ffill()
    peak_px = price_at_or_before(s, peak_date)
    return pd.Series((ss.to_numpy() / peak_px - 1.0) * 100.0, index=(ss.index - peak_date).days)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> list[float]:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return [float(np.interp(q, cdf, values)) for q in qs]


def component_rows(components: list[ModelComponent]) -> pd.DataFrame:
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


def sample_component(c: ModelComponent, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    date_mu = c.bottom_date.toordinal()
    price_mu = c.bottom_price
    dates = rng.normal(date_mu, c.sigma_days, n)
    price_sigma_log = np.log1p(c.sigma_price_pct)
    prices = rng.lognormal(np.log(price_mu), price_sigma_log, n)
    return dates, prices


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    s = load_price_series()
    latest_date = s.index.max().normalize()
    latest_price = float(s.loc[latest_date])

    post_2024 = s.loc[HALVINGS["2024"] : latest_date]
    peak_2025_date = post_2024.idxmax().normalize()
    peak_2025_price = float(post_2024.max())
    current_day_after_peak = int((latest_date - peak_2025_date).days)
    current_drawdown = latest_price / peak_2025_price - 1.0

    hist_rows = []
    for cycle, peak_date in FIXED_PEAKS.items():
        next_halving = HALVINGS["2020"] if cycle == "2017" else HALVINGS["2024"]
        c = rel_curve(s, peak_date, next_halving - pd.Timedelta(days=1))
        bottom_day = int(c.idxmin())
        bottom_date = peak_date + pd.Timedelta(days=bottom_day)
        peak_price = price_at_or_before(s, peak_date)
        bottom_price = float(s.loc[bottom_date])
        hist_rows.append(
            {
                "cycle": cycle,
                "peak_date": peak_date,
                "peak_price": peak_price,
                "bottom_date": bottom_date,
                "bottom_price": bottom_price,
                "days_peak_to_bottom": bottom_day,
                "bottom_to_peak_ratio": bottom_price / peak_price,
                "drawdown_pct": bottom_price / peak_price - 1.0,
            }
        )
    hist = pd.DataFrame(hist_rows)
    hist.to_csv(TABLE_DIR / "historical_peak_to_bottom.csv", index=False)

    bottom_to_next_halving_days = np.array(
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

    next_halving_interval_reg = float(np.polyval(np.polyfit([1, 2, 3], halving_intervals, 1), 4))
    next_halving_date_reg = HALVINGS["2024"] + pd.Timedelta(days=round(next_halving_interval_reg))
    next_halving_date_conservative = HALVINGS["2024"] + pd.Timedelta(days=1440)
    bottom_to_halving_mean = float(bottom_to_next_halving_days.mean())
    bottom_to_halving_std = float(bottom_to_next_halving_days.std(ddof=0))

    days_peak_to_bottom = hist["days_peak_to_bottom"].to_numpy(dtype=float)
    ratios = hist["bottom_to_peak_ratio"].to_numpy(dtype=float)
    trend_ratio = float(np.polyval(np.polyfit([1, 2], ratios, 1), 3))
    trend_ratio = float(np.clip(trend_ratio, 0.25, 0.42))
    old_core_price = weighted_mean(
        np.array([x[1] for x in OLD_CORE_BOTTOM_MODELS]),
        np.array([x[2] for x in OLD_CORE_BOTTOM_MODELS]),
    )
    old_core_ratio = old_core_price / peak_2025_price
    blended_bottom_ratio = 0.55 * trend_ratio + 0.45 * old_core_ratio

    components = [
        ModelComponent(
            name="peak_to_bottom_time_mean",
            bottom_date=peak_2025_date + pd.Timedelta(days=round(days_peak_to_bottom.mean())),
            bottom_price=peak_2025_price * trend_ratio,
            sigma_days=max(18.0, float(days_peak_to_bottom.std(ddof=0)) + 14.0),
            sigma_price_pct=0.14,
            weight=1.25,
            note="2017/2021 peak-to-bottom days, price uses contracting bottom/peak ratio trend",
        ),
        ModelComponent(
            name="bottom_to_next_halving_regressed",
            bottom_date=next_halving_date_reg - pd.Timedelta(days=round(bottom_to_halving_mean)),
            bottom_price=peak_2025_price * blended_bottom_ratio,
            sigma_days=max(42.0, bottom_to_halving_std + 30.0),
            sigma_price_pct=0.15,
            weight=0.75,
            note="estimate next halving interval by regression; contributes late timing, price uses trend/core blended ratio",
        ),
        ModelComponent(
            name="bottom_to_next_halving_conservative",
            bottom_date=next_halving_date_conservative - pd.Timedelta(days=round(bottom_to_halving_mean)),
            bottom_price=peak_2025_price * trend_ratio,
            sigma_days=max(26.0, bottom_to_halving_std + 20.0),
            sigma_price_pct=0.14,
            weight=0.90,
            note="assume next halving interval stays near 2020-2024 interval",
        ),
        ModelComponent(
            name="old_core_bottom_price_anchor",
            bottom_date=peak_2025_date + pd.Timedelta(days=round(days_peak_to_bottom.mean() + 7)),
            bottom_price=old_core_price,
            sigma_days=42.0,
            sigma_price_pct=0.12,
            weight=1.00,
            note="reuse prior core product bottom ensemble as price anchor, updated with actual 2025 peak timing",
        ),
    ]

    comp_df = component_rows(components)
    comp_df.to_csv(TABLE_DIR / "model_components.csv", index=False)

    rng = np.random.default_rng(20260601)
    all_dates = []
    all_prices = []
    all_weights = []
    for c in components:
        dates, prices = sample_component(c, 40000, rng)
        all_dates.append(dates)
        all_prices.append(prices)
        all_weights.append(np.full_like(dates, c.weight, dtype=float))

    date_samples = np.concatenate(all_dates)
    price_samples = np.concatenate(all_prices)
    weights = np.concatenate(all_weights)

    date_q = weighted_quantile(date_samples, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    price_q = weighted_quantile(price_samples, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    date_center_ord = round(weighted_mean(date_samples, weights))
    price_center = weighted_mean(price_samples, weights)
    date_center = pd.Timestamp.fromordinal(date_center_ord)

    monthly = pd.DataFrame(
        {
            "date": [pd.Timestamp.fromordinal(round(x)) for x in date_samples],
            "weight": weights,
        }
    )
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_prob = monthly.groupby("month")["weight"].sum().reset_index()
    monthly_prob["probability"] = monthly_prob["weight"] / monthly_prob["weight"].sum()
    monthly_prob = monthly_prob.drop(columns=["weight"]).sort_values("month")
    monthly_prob.to_csv(TABLE_DIR / "monthly_bottom_probability.csv", index=False)

    path_rows = []
    for _, row in hist.iterrows():
        c = rel_curve(s, row["peak_date"], row["bottom_date"])
        for rel_day, drawdown in c.items():
            path_rows.append(
                {
                    "cycle": row["cycle"],
                    "rel_day": int(rel_day),
                    "drawdown_pct": float(drawdown),
                    "price_if_replayed_from_2025_peak": peak_2025_price * (1.0 + float(drawdown) / 100.0),
                }
            )
    obs = rel_curve(s, peak_2025_date, latest_date)
    for rel_day, drawdown in obs.items():
        path_rows.append(
            {
                "cycle": "2025_observed",
                "rel_day": int(rel_day),
                "drawdown_pct": float(drawdown),
                "price_if_replayed_from_2025_peak": float(s.loc[peak_2025_date + pd.Timedelta(days=int(rel_day))]),
            }
        )
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(TABLE_DIR / "post_peak_paths.csv", index=False)

    plt.figure(figsize=(12, 6.5))
    for cycle, color in [("2017", "#27ae60"), ("2021", "#e74c3c"), ("2025_observed", "#1a5276")]:
        part = path_df[path_df["cycle"] == cycle]
        lw = 2.7 if cycle == "2025_observed" else 1.7
        alpha = 1.0 if cycle == "2025_observed" else 0.45
        plt.plot(part["rel_day"], part["drawdown_pct"], label=cycle, color=color, linewidth=lw, alpha=alpha)
    plt.axvline(current_day_after_peak, color="#1a5276", linestyle="--", alpha=0.8)
    for q in date_q:
        rel = int(round(q)) - peak_2025_date.toordinal()
        plt.axvline(rel, color="gray", linestyle=":", alpha=0.45)
    plt.title("Post-peak drawdown paths and bottom timing band")
    plt.xlabel("days after 2025 actual peak")
    plt.ylabel("drawdown from peak (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "post_peak_drawdown_paths.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    plt.hist(
        [pd.Timestamp.fromordinal(round(x)) for x in date_samples],
        bins=42,
        weights=weights,
        color="#4c78a8",
        alpha=0.78,
    )
    plt.axvline(date_center, color="#111111", linewidth=2, label=f"center {date_center.date()}")
    plt.axvspan(
        pd.Timestamp.fromordinal(round(date_q[0])),
        pd.Timestamp.fromordinal(round(date_q[-1])),
        color="#f58518",
        alpha=0.20,
        label="10-90% band",
    )
    plt.title("Bottom date probability distribution")
    plt.xlabel("date")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_date_distribution.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    plt.hist(price_samples, bins=55, weights=weights, color="#54a24b", alpha=0.78)
    plt.axvline(price_center, color="#111111", linewidth=2, label=f"center ${price_center:,.0f}")
    plt.axvspan(price_q[0], price_q[-1], color="#f58518", alpha=0.20, label="10-90% band")
    plt.axvline(latest_price, color="#1a5276", linestyle="--", label=f"latest ${latest_price:,.0f}")
    plt.title("Bottom price probability distribution")
    plt.xlabel("BTC price")
    plt.ylabel("weighted samples")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "bottom_price_distribution.png", dpi=170)
    plt.close()

    summary = {
        "generated": "2026-06-01",
        "data_file": str(DATA_FILE),
        "data_range": [s.index.min().date().isoformat(), latest_date.date().isoformat()],
        "latest": {"date": latest_date.date().isoformat(), "price": latest_price},
        "actual_2025_peak": {
            "date": peak_2025_date.date().isoformat(),
            "price": peak_2025_price,
            "current_day_after_peak": current_day_after_peak,
            "current_drawdown_pct": current_drawdown * 100.0,
        },
        "next_halving_estimates": {
            "regressed_interval_days": next_halving_interval_reg,
            "regressed_halving_date": next_halving_date_reg.date().isoformat(),
            "conservative_halving_date": next_halving_date_conservative.date().isoformat(),
            "bottom_to_next_halving_days_mean": bottom_to_halving_mean,
            "bottom_to_next_halving_days_std": bottom_to_halving_std,
        },
        "bottom_forecast": {
            "center_date": date_center.date().isoformat(),
            "date_p10": pd.Timestamp.fromordinal(round(date_q[0])).date().isoformat(),
            "date_p25": pd.Timestamp.fromordinal(round(date_q[1])).date().isoformat(),
            "date_p50": pd.Timestamp.fromordinal(round(date_q[2])).date().isoformat(),
            "date_p75": pd.Timestamp.fromordinal(round(date_q[3])).date().isoformat(),
            "date_p90": pd.Timestamp.fromordinal(round(date_q[4])).date().isoformat(),
            "center_price": price_center,
            "price_p10": price_q[0],
            "price_p25": price_q[1],
            "price_p50": price_q[2],
            "price_p75": price_q[3],
            "price_p90": price_q[4],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# BTC Bottom Model V1",
        "",
        f"Generated: 2026-06-01",
        f"Data: {s.index.min().date()} to {latest_date.date()}",
        f"Latest BTC: ${latest_price:,.2f} on {latest_date.date()}",
        "",
        "## Peak anchor",
        f"- Actual 2025 cycle high since 2024 halving: {peak_2025_date.date()} at ${peak_2025_price:,.2f}",
        f"- Current position: day {current_day_after_peak} after peak, drawdown {current_drawdown * 100:.2f}%",
        "",
        "## New bottom forecast",
        f"- Center date: {date_center.date()}",
        f"- Date band 10-90%: {pd.Timestamp.fromordinal(round(date_q[0])).date()} to {pd.Timestamp.fromordinal(round(date_q[-1])).date()}",
        f"- Date band 25-75%: {pd.Timestamp.fromordinal(round(date_q[1])).date()} to {pd.Timestamp.fromordinal(round(date_q[3])).date()}",
        f"- Center price: ${price_center:,.0f}",
        f"- Price band 10-90%: ${price_q[0]:,.0f} to ${price_q[-1]:,.0f}",
        f"- Price band 25-75%: ${price_q[1]:,.0f} to ${price_q[3]:,.0f}",
        "",
        "## Model components",
        comp_df.to_string(index=False),
        "",
        "## Interpretation",
        "This model treats 2025-10-05 as the real peak. It combines peak-to-bottom timing, bottom-to-next-halving timing, historical bottom/peak ratio decay, and the old core bottom-price ensemble.",
        "The result should be read as a probabilistic bottom zone, not a single exact day.",
    ]
    (OUT_DIR / "bottom_model_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
