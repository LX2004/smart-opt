import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = {
    "BASDDPG": BASE_DIR / "BASDDPG_result",
    "MCSA_SPDDPG": BASE_DIR / "MCSA_SPDDPG_result",
    "SAC": BASE_DIR / "SAC_result",
    "DDPG": BASE_DIR / "DDPG_result",
    "TD3": BASE_DIR / "TD3_result",
}
DEFAULT_FIGURE_DIR = BASE_DIR / "Figures"

ALGO_COLORS = {
    "BASDDPG": "#1f77b4",
    "MCSA_SPDDPG": "#9467bd",
    "SAC": "#ff7f0e",
    "DDPG": "#2ca02c",
    "TD3": "#d62728",
}

SIGNED_COMPONENTS_SUM = [
    ("curtail_penalty_sum", "Curtail penalty"),
    ("comfort_penalty_sum", "Comfort penalty"),
    ("h2_tank_penalty_sum", "H2 tank penalty"),
    ("carbon_penalty_sum", "Carbon penalty"),
    ("bio_starvation_penalty_sum", "Bio starvation penalty"),
    ("biomass_reward_signed_sum", "Biomass reward"),
    ("harvest_reward_signed_sum", "SCP harvest reward"),
]

SIGNED_COMPONENTS_MEAN = [
    ("curtail_penalty_mean", "Curtail penalty / h"),
    ("comfort_penalty_mean", "Comfort penalty / h"),
    ("h2_tank_penalty_mean", "H2 tank penalty / h"),
    ("carbon_penalty_mean", "Carbon penalty / h"),
    ("bio_starvation_penalty_mean", "Bio starvation penalty / h"),
    ("biomass_reward_signed_mean", "Biomass reward / h"),
    ("harvest_reward_signed_mean", "SCP harvest reward / h"),
]

RAW_COMPONENTS_SUM = [
    ("step_cost_sum", "Step cost"),
    ("curtail_cost_sum", "Curtail cost"),
    ("comfort_cost_weighted_sum", "Comfort cost"),
    ("h2_tank_violation_cost_sum", "H2 tank violation cost"),
    ("carbon_cost_sum", "Carbon cost"),
    ("bio_starvation_cost_sum", "Bio starvation cost"),
    ("biomass_reward_component_sum", "Biomass reward component"),
    ("harvest_reward_component_sum", "Harvest reward component"),
]

SYSTEM_METRICS = [
    ("grid_purchase_mwh_sum", "Grid purchase, MWh"),
    ("grid_co2_emission_t_sum", "Grid CO2 emission, t"),
    ("bio_co2_absorption_t_sum", "Bio CO2 absorption, t"),
    ("net_co2_emission_t_sum", "Net CO2 emission, t"),
    ("final_h2_tank_mol", "Final H2 tank, mol"),
    ("final_battery_soc_mwh", "Final battery SOC, MWh"),
    ("mean_bio_H2_growth_fraction", "Mean bio H2 growth fraction"),
    ("mean_bio_starvation_level", "Mean bio starvation level"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize test-set epoch reward components for microgrid RL results."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where PNG figures will be saved.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional rolling mean window over epochs. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="epoch_reward_components_test.csv",
        help="CSV file name under each result directory.",
    )
    return parser.parse_args()


def read_result_csvs(csv_name):
    frames = {}
    for algo, result_dir in DEFAULT_RESULTS.items():
        csv_path = result_dir / csv_name
        if not csv_path.exists():
            print(f"[WARN] Missing {algo} CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[WARN] Empty {algo} CSV: {csv_path}")
            continue
        for col in df.columns:
            if col != "split":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
        add_signed_reward_columns(df)
        frames[algo] = df
        print(f"[OK] {algo}: {csv_path} ({len(df)} epochs)")
    if not frames:
        raise FileNotFoundError("No usable epoch_reward_components_test.csv files were found.")
    return frames


def add_signed_reward_columns(df):
    # Signed contribution convention: penalties are negative reward terms;
    # biomass and harvest components are positive reward terms.
    mapping = {
        "curtail_penalty": "curtail_cost",
        "comfort_penalty": "comfort_cost_weighted",
        "h2_tank_penalty": "h2_tank_violation_cost",
        "carbon_penalty": "carbon_cost",
        "bio_starvation_penalty": "bio_starvation_cost",
    }
    for signed_name, raw_name in mapping.items():
        for suffix in ["sum", "mean"]:
            raw_col = f"{raw_name}_{suffix}"
            if raw_col in df:
                df[f"{signed_name}_{suffix}"] = -df[raw_col]
    for suffix in ["sum", "mean"]:
        biomass_col = f"biomass_reward_component_{suffix}"
        harvest_col = f"harvest_reward_component_{suffix}"
        if biomass_col in df:
            df[f"biomass_reward_signed_{suffix}"] = df[biomass_col]
        if harvest_col in df:
            df[f"harvest_reward_signed_{suffix}"] = df[harvest_col]


def smooth_series(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(ax, frames, column, title, ylabel, smooth_window):
    plotted = False
    for algo, df in frames.items():
        if column not in df.columns:
            continue
        y = smooth_series(df[column], smooth_window)
        ax.plot(
            df["epoch"],
            y,
            label=algo,
            linewidth=1.8,
            color=ALGO_COLORS.get(algo),
        )
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, f"Missing column: {column}", ha="center", va="center")


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}")


def plot_total_rewards(frames, figure_dir, smooth_window):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Test Reward vs Epoch", fontsize=15)
    plot_metric(axes[0], frames, "total_reward", "Total reward on full test set", "Total reward", smooth_window)
    plot_metric(axes[1], frames, "mean_reward", "Mean reward per hour on full test set", "Reward / h", smooth_window)
    save_figure(fig, figure_dir / "test_total_and_mean_reward.png")


def plot_component_grid(frames, components, figure_dir, filename, title, ylabel, smooth_window):
    n = len(components)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for ax, (column, label) in zip(flat_axes, components):
        plot_metric(ax, frames, column, label, ylabel, smooth_window)
    for ax in flat_axes[n:]:
        ax.axis("off")
    save_figure(fig, figure_dir / filename)


def plot_reward_reconstruction(frames, figure_dir, smooth_window):
    signed_cols = [col for col, _ in SIGNED_COMPONENTS_SUM]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Signed Reward Contribution Check", fontsize=15)
    for algo, df in frames.items():
        available = [col for col in signed_cols if col in df.columns]
        if not available:
            continue
        reconstructed = df[available].sum(axis=1)
        axes[0].plot(
            df["epoch"],
            smooth_series(reconstructed, smooth_window),
            label=f"{algo} signed components",
            color=ALGO_COLORS.get(algo),
            linewidth=1.6,
        )
        axes[1].plot(
            df["epoch"],
            smooth_series(df["total_reward"] - reconstructed, smooth_window),
            label=algo,
            color=ALGO_COLORS.get(algo),
            linewidth=1.6,
        )
    axes[0].set_ylabel("Signed contribution sum")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Total reward - shown components")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False)
    save_figure(fig, figure_dir / "test_signed_reward_reconstruction.png")


def main():
    args = parse_args()
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    frames = read_result_csvs(args.csv_name)

    plot_total_rewards(frames, figure_dir, args.smooth_window)
    plot_component_grid(
        frames,
        SIGNED_COMPONENTS_SUM,
        figure_dir,
        "test_signed_reward_components_sum.png",
        "Signed Test Reward Components by Epoch (Sum)",
        "Signed reward contribution",
        args.smooth_window,
    )
    plot_component_grid(
        frames,
        SIGNED_COMPONENTS_MEAN,
        figure_dir,
        "test_signed_reward_components_mean.png",
        "Signed Test Reward Components by Epoch (Mean per Hour)",
        "Signed reward contribution / h",
        args.smooth_window,
    )
    plot_component_grid(
        frames,
        RAW_COMPONENTS_SUM,
        figure_dir,
        "test_raw_cost_reward_components_sum.png",
        "Raw Test Cost and Reward Components by Epoch (Sum)",
        "Raw component value",
        args.smooth_window,
    )
    plot_component_grid(
        frames,
        SYSTEM_METRICS,
        figure_dir,
        "test_system_metrics_by_epoch.png",
        "System and Carbon Metrics on Test Set by Epoch",
        "Metric value",
        args.smooth_window,
    )
    plot_reward_reconstruction(frames, figure_dir, args.smooth_window)


if __name__ == "__main__":
    main()
