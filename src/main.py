"""
PJM Congestion & Constraint Playbook (Desk-Style)

Run:
  python src/main.py --days 60 --market day_ahead --outdir outputs
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from gridstatus import PJM
except Exception as e:
    raise SystemExit(
        "Missing dependency gridstatus. Install with: pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

UTC = timezone.utc


@dataclass
class Config:
    days: int
    market: str
    outdir: str
    ref_location: str | None
    max_locations: int
    seed: int = 7


def ensure_dirs(outdir: str) -> dict:
    paths = {
        "root": outdir,
        "data": os.path.join(outdir, "data"),
        "tables": os.path.join(outdir, "tables"),
        "fig": os.path.join(outdir, "figures"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def utc_date_range(days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(datetime.now(tz=UTC).date())
    start = end - pd.Timedelta(days=days)
    return start, end


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    time_col = "time" if "time" in cols else None

    loc_col = None
    for c in ["location", "pnode", "node", "pricing_node"]:
        if c in cols:
            loc_col = c
            break

    price_col = None
    for c in ["lmp", "total_lmp", "price"]:
        if c in cols:
            price_col = c
            break

    cong_col = None
    for c in ["congestion", "congestion_component", "mcc", "marginal_cost_congestion"]:
        if c in cols:
            cong_col = c
            break

    loss_col = None
    for c in ["loss", "mcl", "marginal_cost_losses"]:
        if c in cols:
            loss_col = c
            break

    energy_col = None
    for c in ["energy", "mec", "marginal_energy_component"]:
        if c in cols:
            energy_col = c
            break

    if not (time_col and loc_col and price_col):
        raise ValueError(f"Could not infer required columns. Found columns: {sorted(df.columns)}")

    return {"time": time_col, "loc": loc_col, "price": price_col, "cong": cong_col, "loss": loss_col, "energy": energy_col}


def download_lmp(cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    iso = PJM()
    market = cfg.market.lower().strip()
    if market not in {"day_ahead", "real_time"}:
        raise ValueError("market must be 'day_ahead' or 'real_time'")

    df = iso.get_lmp(date=start, end=end, market=market)
    df = normalize_cols(df)
    m = infer_columns(df)

    keep = [m["time"], m["loc"], m["price"]]
    for k in ["cong", "loss", "energy"]:
        if m[k]:
            keep.append(m[k])
    df = df[keep].copy()

    rename = {m["time"]: "time", m["loc"]: "location", m["price"]: "lmp"}
    if m["cong"]:
        rename[m["cong"]] = "congestion"
    if m["loss"]:
        rename[m["loss"]] = "loss"
    if m["energy"]:
        rename[m["energy"]] = "energy"
    df = df.rename(columns=rename)

    df = df.dropna(subset=["time", "location", "lmp"]).sort_values("time").reset_index(drop=True)

    top_locs = df["location"].value_counts().head(cfg.max_locations).index.tolist()
    df = df[df["location"].isin(top_locs)].copy()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    return df


def congestion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "congestion" in df.columns:
        df["cong_value"] = df["congestion"].astype(float)
    else:
        med = df.groupby("time")["lmp"].median().rename("median_lmp")
        df = df.merge(med, on="time", how="left")
        df["cong_value"] = (df["lmp"] - df["median_lmp"]).astype(float)
    df["cong_abs"] = df["cong_value"].abs()
    return df


def build_location_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("location", as_index=False).agg(
        avg_cong=("cong_value", "mean"),
        avg_abs_cong=("cong_abs", "mean"),
        p95_abs_cong=("cong_abs", lambda x: float(np.nanpercentile(x, 95))),
        std_cong=("cong_value", "std"),
        n=("cong_value", "size"),
    )
    g["tail_score"] = g["p95_abs_cong"] * 0.7 + g["avg_abs_cong"] * 0.3
    return g.sort_values("tail_score", ascending=False).reset_index(drop=True)


def basis_playbook(df: pd.DataFrame, ref_location: str | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    locs = sorted(df["location"].unique().tolist())
    ref = ref_location if (ref_location in locs) else (ref_location or locs[0])

    pivot = df.pivot_table(index="time", columns="location", values="lmp", aggfunc="mean")

    basis = pivot.sub(pivot[ref], axis=0).drop(columns=[ref], errors="ignore")
    basis_long = basis.reset_index().melt(id_vars=["time"], var_name="location", value_name="basis_vs_ref")
    basis_long["basis_abs"] = basis_long["basis_vs_ref"].abs()

    pair_rows = []
    cols = list(pivot.columns)
    for a, b in combinations(cols, 2):
        s = (pivot[a] - pivot[b]).dropna()
        if s.empty:
            continue
        pair_rows.append({
            "pair": f"{a} vs {b}",
            "p95_abs_spread": float(np.nanpercentile(np.abs(s), 95)),
            "avg_abs_spread": float(np.nanmean(np.abs(s))),
            "n": int(s.shape[0]),
        })
    pairs = pd.DataFrame(pair_rows).sort_values("p95_abs_spread", ascending=False).reset_index(drop=True)

    return basis_long, pairs, ref


def seasonality_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hm = df.pivot_table(index="hour", columns="month", values="cong_abs", aggfunc="mean")
    hd = df.pivot_table(index="hour", columns="dow", values="cong_abs", aggfunc="mean")
    return hm, hd


def save_tables(paths: dict, leaderboard: pd.DataFrame, basis_long: pd.DataFrame, pairs: pd.DataFrame, hm: pd.DataFrame, hd: pd.DataFrame) -> None:
    leaderboard.to_csv(os.path.join(paths["tables"], "location_congestion_leaderboard.csv"), index=False)
    basis_long.to_csv(os.path.join(paths["tables"], "basis_vs_ref_long.csv"), index=False)
    pairs.to_csv(os.path.join(paths["tables"], "top_basis_pairs.csv"), index=False)
    hm.to_csv(os.path.join(paths["tables"], "seasonality_hour_by_month.csv"))
    hd.to_csv(os.path.join(paths["tables"], "seasonality_hour_by_dow.csv"))


def plot_heatmap(mat: pd.DataFrame, title: str, path: str) -> None:
    plt.figure(figsize=(10, 5))
    arr = mat.to_numpy()
    plt.imshow(arr, aspect="auto")
    plt.title(title)
    plt.xlabel(mat.columns.name or "column")
    plt.ylabel(mat.index.name or "row")
    plt.colorbar()
    plt.xticks(range(len(mat.columns)), [str(c) for c in mat.columns], rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), [str(i) for i in mat.index])
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_outputs(paths: dict, leaderboard: pd.DataFrame, hm: pd.DataFrame, hd: pd.DataFrame) -> None:
    top = leaderboard.head(15).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(top["location"].astype(str), top["tail_score"].astype(float))
    plt.title("Top locations by congestion tail score (p95 + avg)")
    plt.xlabel("Location")
    plt.ylabel("Tail score")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["fig"], "top_locations_tail_score.png"), dpi=160)
    plt.close()

    plot_heatmap(hm.fillna(0.0), "Seasonality: avg congestion magnitude by hour and month", os.path.join(paths["fig"], "seasonality_hour_by_month.png"))
    plot_heatmap(hd.fillna(0.0), "Seasonality: avg congestion magnitude by hour and day-of-week", os.path.join(paths["fig"], "seasonality_hour_by_dow.png"))


def write_report(cfg: Config, paths: dict, leaderboard: pd.DataFrame, pairs: pd.DataFrame, ref_location: str) -> str:
    top_locs = leaderboard.head(5)[["location", "tail_score", "p95_abs_cong"]].to_dict(orient="records")
    top_pairs = pairs.head(5)[["pair", "p95_abs_spread"]].to_dict(orient="records")

    lines = [
        f"# PJM Congestion & Constraint Playbook ({cfg.market})",
        "",
        "## Reference location",
        f"- {ref_location}",
        "",
        "## Top locations by congestion tail risk",
    ]
    for r in top_locs:
        lines.append(f"- {r['location']}: tail_score={r['tail_score']:.2f}, p95_abs_cong={r['p95_abs_cong']:.2f}")
    lines += ["", "## Top basis blowout pairs (p95 abs spread)"]
    for r in top_pairs:
        lines.append(f"- {r['pair']}: p95_abs_spread={r['p95_abs_spread']:.2f}")
    lines += [
        "",
        "## Interview-ready one-liner",
        "Built a PJM congestion playbook by ranking locations on congestion tail risk and mapping basis blowouts across nodes and hubs, translating repeatable patterns into hedging and positioning intuition.",
        "",
        "## Outputs",
        "- tables/location_congestion_leaderboard.csv",
        "- tables/top_basis_pairs.csv",
        "- figures/seasonality_hour_by_month.png",
        "- figures/seasonality_hour_by_dow.png",
    ]

    report_path = os.path.join(paths["root"], "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--market", type=str, default="day_ahead")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--ref_location", type=str, default=None)
    ap.add_argument("--max_locations", type=int, default=35)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    return Config(days=a.days, market=a.market, outdir=a.outdir, ref_location=a.ref_location, max_locations=a.max_locations, seed=a.seed)


def main() -> None:
    cfg = parse_args()
    paths = ensure_dirs(cfg.outdir)

    start, end = utc_date_range(cfg.days)
    raw = download_lmp(cfg, start=start, end=end)
    raw = add_time_features(raw)
    raw = congestion_metrics(raw)

    raw_path = os.path.join(paths["data"], f"pjm_lmp_{cfg.market}_{cfg.days}d.parquet")
    raw.to_parquet(raw_path, index=False)

    leaderboard = build_location_leaderboard(raw)
    basis_long, pairs, ref = basis_playbook(raw, cfg.ref_location)
    hm, hd = seasonality_tables(raw)

    save_tables(paths, leaderboard, basis_long, pairs, hm, hd)
    plot_outputs(paths, leaderboard, hm, hd)
    report_path = write_report(cfg, paths, leaderboard, pairs, ref)

    print("Saved raw parquet:", raw_path)
    print("Saved report:", report_path)
    print("Top locations (tail_score):")
    print(leaderboard.head(10).to_string(index=False))
    print("\nTop basis pairs (p95 abs spread):")
    print(pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
