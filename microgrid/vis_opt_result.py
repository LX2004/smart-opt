import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = {
    "DDPG-Safety": (BASE_DIR / "DDPG_safety", BASE_DIR / "BASDDPG_result"),
    "DDPG": (BASE_DIR / "DDPG_result", BASE_DIR / "DDPG_results"),
    "MCSA_SPDDPG": (BASE_DIR / "MCSA_SPDDPG_result",),
    "SAC-Safety": (BASE_DIR / "SAC_safety",),
    "SAC": (BASE_DIR / "SAC_result", BASE_DIR / "SAC"),
    "TD3-Safety": (BASE_DIR / "TD3_safety",),
    "TD3": (BASE_DIR / "TD3_result", BASE_DIR / "TD3"),
}
RESULT_GROUPS = {
    "safety": {
        "label": "With Safety Constraints",
        "algorithms": ["DDPG-Safety", "MCSA_SPDDPG", "SAC-Safety", "TD3-Safety"],
    },
    "no_safety": {
        "label": "Without Safety Constraints",
        "algorithms": ["DDPG", "SAC", "TD3"],
    },
}
DEFAULT_FIGURE_DIR = BASE_DIR / "Figures"
DEFAULT_SCP_PRICE_YUAN_PER_KG_PROTEIN = 12.0
DEFAULT_MAX_EPOCH = 500

ALGO_COLORS = {
    "DDPG-Safety": "#1f77b4",
    "DDPG": "#1f77b4",
    "MCSA_SPDDPG": "#9467bd",
    "SAC-Safety": "#ff7f0e",
    "SAC": "#ff7f0e",
    "TD3-Safety": "#2ca02c",
    "TD3": "#2ca02c",
}
ALGO_LINESTYLES = {
    "DDPG": "--",
    "SAC": "--",
    "TD3": "--",
}
VIOLATION_COST_TO_COUNT = {
    "h2_violation_cost_yuan_sum": 1000.0,
    "battery_violation_cost_yuan_sum": 1000.0,
    "comfort_violation_cost_yuan_sum": 1000.0,
    "bio_h2_violation_cost_yuan_sum": 1000.0,
    "bio_forced_replacement_cost_yuan_sum": 2000.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grouped microgrid optimization results."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where figures and summary CSV files will be saved.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling mean window for reward curves. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=DEFAULT_MAX_EPOCH,
        help="Only visualize epochs up to and including this value. Use 0 for all epochs.",
    )
    parser.add_argument(
        "--scp-price",
        type=float,
        default=DEFAULT_SCP_PRICE_YUAN_PER_KG_PROTEIN,
        help="SCP protein price used to convert SCP revenue to kg protein.",
    )
    parser.add_argument(
        "--epoch-csv-name",
        type=str,
        default="epoch_reward_components_test.csv",
        help="Epoch-level test CSV file name under each result directory.",
    )
    return parser.parse_args()


def choose_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def choose_result_dir(result_dirs, csv_name):
    for result_dir in result_dirs:
        if (result_dir / csv_name).exists():
            return result_dir
    return None


def limit_epoch_range(df, max_epoch):
    if max_epoch and max_epoch > 0:
        return df[df["epoch"] <= max_epoch].reset_index(drop=True)
    return df.reset_index(drop=True)


def read_epoch_frames(csv_name, scp_price, max_epoch):
    frames = {}
    for algo, result_dirs in DEFAULT_RESULTS.items():
        result_dir = choose_result_dir(result_dirs, csv_name)
        if result_dir is None:
            checked = ", ".join(str(result_dir / csv_name) for result_dir in result_dirs)
            print(f"[WARN] Missing {algo} CSV. Checked: {checked}")
            continue

        csv_path = result_dir / csv_name
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[WARN] Empty {algo} CSV: {csv_path}")
            continue
        for col in df.columns:
            if col != "split":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
        df = limit_epoch_range(df, max_epoch)
        if df.empty:
            print(f"[WARN] No {algo} rows remain after max_epoch={max_epoch}: {csv_path}")
            continue

        add_epoch_derived_columns(df, scp_price)
        frames[algo] = df
        print(f"[OK] {algo}: {csv_path} ({len(df)} epochs used for stats/plots)")
    if not frames:
        raise FileNotFoundError(f"No usable {csv_name} files were found.")
    return frames


def ensure_numeric_column(df, column):
    if column not in df:
        df[column] = 0.0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def add_epoch_derived_columns(df, scp_price):
    safe_price = scp_price if scp_price > 0 else DEFAULT_SCP_PRICE_YUAN_PER_KG_PROTEIN
    has_curtail_kwh = "curtail_kwh_sum" in df
    has_curtail_mwh = "curtail_mwh_sum" in df
    has_curtail_cost = "curtail_cost_yuan_sum" in df
    for column in [
        "scp_growth_revenue_yuan_sum",
        "scp_harvest_revenue_yuan_sum",
        "bio_co2_absorption_t_sum",
        "grid_co2_emission_t_sum",
        "net_co2_emission_t_sum",
        "curtail_kwh_sum",
        "curtail_mwh_sum",
        "curtail_cost_yuan_sum",
    ]:
        ensure_numeric_column(df, column)
    for column in VIOLATION_COST_TO_COUNT:
        ensure_numeric_column(df, column)

    if not has_curtail_kwh:
        if has_curtail_mwh:
            df["curtail_kwh_sum"] = df["curtail_mwh_sum"] * 1000.0
        elif has_curtail_cost:
            # Current reward uses 1 yuan/kWh curtailment cost, so old files with
            # only cost can still be visualized as kWh.
            df["curtail_kwh_sum"] = df["curtail_cost_yuan_sum"]
    df["curtail_mwh_sum"] = df["curtail_kwh_sum"] / 1000.0

    df["scp_growth_kg_protein_sum"] = df["scp_growth_revenue_yuan_sum"] / safe_price
    df["scp_harvest_kg_protein_sum"] = df["scp_harvest_revenue_yuan_sum"] / safe_price
    df["scp_total_kg_protein_sum"] = (
        df["scp_growth_kg_protein_sum"] + df["scp_harvest_kg_protein_sum"]
    )
    # Positive means the bio-load absorbed more CO2 than external grid purchase emitted.
    df["net_co2_absorption_t_sum"] = (
        df["bio_co2_absorption_t_sum"] - df["grid_co2_emission_t_sum"]
    )
    df["total_violation_count"] = 0.0
    for cost_col, unit_penalty in VIOLATION_COST_TO_COUNT.items():
        count_col = cost_col.replace("_cost_yuan_sum", "_count")
        df[count_col] = df[cost_col] / unit_penalty
        df["total_violation_count"] += df[count_col]


def smooth_series(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def select_group_frames(frames, algo_names):
    return {algo: frames[algo] for algo in algo_names if algo in frames}


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}")


def best_reward_rows(frames):
    # Frames are already clipped by max_epoch in read_epoch_frames, so all
    # max/min-derived summaries below are calculated only within that window.
    rows = {}
    for algo, df in frames.items():
        rows[algo] = df.loc[df["total_reward"].idxmax()]
    return rows


def plot_reward_curve(frames, figure_dir, prefix, group_label, smooth_window):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo, df in frames.items():
        y = smooth_series(df["total_reward"], smooth_window)
        ax.plot(
            df["epoch"],
            y,
            label=algo,
            color=ALGO_COLORS.get(algo),
            linestyle=ALGO_LINESTYLES.get(algo, "-"),
            linewidth=1.9,
        )
    ax.set_title(f"{group_label}: Test Reward Trend")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / f"{prefix}_reward_vs_epoch.png")


def plot_best_reward_metric_bars(frames, figure_dir, prefix, group_label):
    selected = best_reward_rows(frames)
    algos = list(selected.keys())
    colors = [ALGO_COLORS.get(algo, "#666666") for algo in algos]
    co2_absorption = [selected[algo]["bio_co2_absorption_t_sum"] for algo in algos]
    net_co2_absorption = [selected[algo]["net_co2_absorption_t_sum"] for algo in algos]
    scp_production = [selected[algo]["scp_total_kg_protein_sum"] for algo in algos]
    curtailment = [selected[algo]["curtail_mwh_sum"] for algo in algos]
    violation_counts = [selected[algo]["total_violation_count"] for algo in algos]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    specs = [
        (axes[0, 0], co2_absorption, "CO2 absorption", "t"),
        (axes[0, 1], net_co2_absorption, "Net CO2 absorption", "t"),
        (axes[0, 2], curtailment, "Curtailment", "MWh"),
        (axes[1, 0], scp_production, "SCP production", "kg protein"),
        (axes[1, 1], violation_counts, "Violation count", "count"),
    ]
    axes[1, 2].axis("off")
    for ax, values, title, ylabel in specs:
        bars = ax.bar(algos, values, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle(f"{group_label}: Metrics at the Best-Reward Epoch")
    save_figure(fig, figure_dir / f"{prefix}_best_reward_epoch_metrics.png")


def write_best_reward_summary(frames, figure_dir, prefix, group_label):
    rows = []
    for algo, row in best_reward_rows(frames).items():
        rows.append(
            {
                "group": group_label,
                "method": algo,
                "best_reward_epoch": int(row["epoch"]),
                "best_test_reward": float(row["total_reward"]),
                "co2_absorption_t_at_best_reward": float(row["bio_co2_absorption_t_sum"]),
                "grid_co2_emission_t_at_best_reward": float(row["grid_co2_emission_t_sum"]),
                "net_co2_absorption_t_at_best_reward": float(row["net_co2_absorption_t_sum"]),
                "curtail_mwh_at_best_reward": float(row["curtail_mwh_sum"]),
                "curtail_kwh_at_best_reward": float(row["curtail_kwh_sum"]),
                "curtail_cost_yuan_at_best_reward": float(row["curtail_cost_yuan_sum"]),
                "scp_kg_protein_at_best_reward": float(row["scp_total_kg_protein_sum"]),
                "total_violation_count_at_best_reward": float(row["total_violation_count"]),
                "h2_violation_count_at_best_reward": float(row["h2_violation_count"]),
                "battery_violation_count_at_best_reward": float(row["battery_violation_count"]),
                "comfort_violation_count_at_best_reward": float(row["comfort_violation_count"]),
                "bio_h2_violation_count_at_best_reward": float(row["bio_h2_violation_count"]),
                "bio_forced_replacement_count_at_best_reward": float(row["bio_forced_replacement_count"]),
            }
        )
    path = figure_dir / f"{prefix}_best_reward_epoch_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def plot_group_figures(frames, figure_dir, prefix, group_label, smooth_window):
    plot_reward_curve(frames, figure_dir, prefix, group_label, smooth_window)
    plot_best_reward_metric_bars(frames, figure_dir, prefix, group_label)
    write_best_reward_summary(frames, figure_dir, prefix, group_label)


def read_mcsa_test_details():
    result_dir = choose_result_dir(DEFAULT_RESULTS["MCSA_SPDDPG"], "test_timeseries_details.csv")
    if result_dir is None:
        raise FileNotFoundError("Cannot find MCSA_SPDDPG test_timeseries_details.csv")

    details_path = result_dir / "test_timeseries_details.csv"
    df = pd.read_csv(details_path)
    for col in df.columns:
        if col != "bio_replacement_reason":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    npz_path = choose_existing_path(
        [
            result_dir / "mcsa_spddpg_microgrid_test_results.npz",
            result_dir / "mcsa_spddpg_microgrid_validation_results.npz",
        ]
    )
    if npz_path is not None:
        data = np.load(npz_path)
        if "t_out" in data and len(data["t_out"]) >= len(df):
            df["T_out_C"] = data["t_out"][: len(df)]
    if "T_out_C" not in df:
        print("[WARN] MCSA outdoor temperature t_out was not found; temperature plot will omit it.")

    print(f"[OK] MCSA test details: {details_path} ({len(df)} time steps)")
    return df


def series_or_zero(df, column):
    if column in df:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def plot_mcsa_best_test_timeseries(figure_dir):
    df = read_mcsa_test_details()
    time = df["time_step"] if "time_step" in df else pd.Series(range(len(df)))

    scp_kg = series_or_zero(df, "scp_total_available_g_protein") / 1000.0
    cumulative_co2_absorption_t = series_or_zero(df, "co2_absorbed_t").cumsum()
    cumulative_grid_emission_t = series_or_zero(df, "grid_co2_emission_t").cumsum()
    cumulative_net_absorption_t = cumulative_co2_absorption_t - cumulative_grid_emission_t

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("MCSA_SPDDPG Best Model on Test Set")

    axes[0].plot(time, scp_kg, color="#9467bd", linewidth=1.8)
    axes[0].set_ylabel("SCP protein, kg")
    axes[0].set_title("SCP production over time")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(time, cumulative_co2_absorption_t, label="Bio CO2 absorption", color="#2ca02c", linewidth=1.6)
    axes[1].plot(time, cumulative_grid_emission_t, label="Grid CO2 emission", color="#d62728", linewidth=1.4)
    axes[1].plot(time, cumulative_net_absorption_t, label="Net absorption after grid emission", color="#111111", linewidth=1.8)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("CO2, t")
    axes[1].set_title("Cumulative CO2 absorption and net absorption over time")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False, ncols=3)

    axes[2].plot(time, df["T_room_C"], label="Indoor", color="#d62728", linewidth=1.5)
    axes[2].plot(time, df["T_wall_C"], label="Wall", color="#1f77b4", linewidth=1.5)
    if "T_out_C" in df:
        axes[2].plot(time, df["T_out_C"], label="Outdoor", color="#7f7f7f", linewidth=1.3)
    axes[2].set_ylabel("Temperature, C")
    axes[2].set_title("Indoor, wall, and outdoor temperature over time")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].legend(frameon=False, ncols=3)

    axes[3].plot(time, series_or_zero(df, "battery_soc_kwh"), color="#ff7f0e", linewidth=1.7)
    axes[3].set_ylabel("Battery SOC, kWh")
    axes[3].set_title("Battery state of charge over time")
    axes[3].grid(True, linestyle="--", alpha=0.3)

    axes[4].plot(time, series_or_zero(df, "h2_tank_mol"), color="#17becf", linewidth=1.7)
    axes[4].set_xlabel("Test time step, h")
    axes[4].set_ylabel("H2 tank, mol")
    axes[4].set_title("Hydrogen storage over time")
    axes[4].grid(True, linestyle="--", alpha=0.3)

    save_figure(fig, figure_dir / "mcsa_best_test_scp_co2_temperature_storage_timeseries.png")


def main():
    args = parse_args()
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    frames = read_epoch_frames(args.epoch_csv_name, args.scp_price, args.max_epoch)
    epoch_tag = f"first{args.max_epoch}epoch" if args.max_epoch and args.max_epoch > 0 else "allepoch"

    for group_key, group in RESULT_GROUPS.items():
        group_frames = select_group_frames(frames, group["algorithms"])
        if not group_frames:
            print(f"[WARN] No data for group: {group['label']}")
            continue
        plot_group_figures(
            group_frames,
            figure_dir,
            f"{group_key}_{epoch_tag}",
            group["label"],
            args.smooth_window,
        )

    plot_mcsa_best_test_timeseries(figure_dir)


if __name__ == "__main__":
    main()
