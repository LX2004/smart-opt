import argparse
import csv
import os
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from eq import PV, smart_building, wind_turbine
from utils import get_weather_data
from bio_load import HOBHydrogenLoad, HOBLoadConfig, HOBParams


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Configs:
    def __init__(self):
        self.pv_num = 4e5
        self.std_irradiance = 1000
        self.temperature_coeff = -0.004
        self.std_temp = 25
        self.rated_power = 300

        self.hydrogen_storage_pressure_max = 30.0
        self.hydrogen_storage_pressure_min = 2.0
        self.ideal_gas_constant = 8.314
        self.hydrogen_storage_temp = 298.15
        self.hydrogen_tank_vol = 5.0
        self.hydrogen_tank_max_pressure = 35.0
        self.h2_tank_mol_min = 4e3
        self.h2_tank_mol_max = 6e4
        self.h2_tank_num = 40

        self.num_cell = 2e3
        self.area = 1000

        self.ev2fcev_ratio_mol = 200.0
        self.h2_electrolyzer_kwh_per_mol = 0.1135
        self.human_comfort_temp = 24
        self.building_num = 2
        self.wind_turbine_num = 100
        self.source_to_load_ratio = 1.3
        self.ev_to_bio_h2_ratio = 0.2
        self.electric_load_to_bio_electric_ratio = 0.2
        self.grid_emission_factor_tco2_per_mwh = 0.5703
        self.co2_molar_mass_g_per_mol = 44.0095


@dataclass
class SACConfig:
    episodes: int = 120
    batch_size: int = 256
    buffer_size: int = 200000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    hidden_dim: int = 256
    exploration_noise: float = 1.0
    noise_decay: float = 1.0
    min_noise: float = 1.0
    alpha_lr: float = 3e-5
    init_alpha: float = 0.2
    target_entropy: float = -4.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    seed: int = 8
    random_train_window: bool = True
    episode_window_hours: int = 72


@dataclass
class ScenarioData:
    pv: np.ndarray
    wind: np.ndarray
    load: np.ndarray
    ev_demand: np.ndarray
    t_out: np.ndarray
    wind100: np.ndarray
    irradiance: np.ndarray


class FixedStrainHydrogenLoad:
    """HOB hydrogen load with a fixed inoculum for every fermenter batch.

    Each fermenter starts from the same fixed X0_gdw. The active biomass can grow
    within a batch. Replacement is controlled here for the DDPG scheduler by
    low-efficiency and severe-H2-shortfall rules.
    """

    def __init__(
        self,
        fixed_X0_gdw=100.0,
        h2_growth_fraction_cutoff=0.5,
        min_runtime_before_replacement_h=24.0,
        physical_h2_feed_max_mol_h=5000.0,
        severe_h2_shortfall_cutoff_h=6.0,
    ):
        self.fixed_X0_gdw = float(fixed_X0_gdw)
        self.h2_growth_fraction_cutoff = float(h2_growth_fraction_cutoff)
        self.min_runtime_before_replacement_h = float(min_runtime_before_replacement_h)
        self.physical_h2_feed_max_mol_h_value = float(physical_h2_feed_max_mol_h)
        self.severe_h2_shortfall_cutoff_h = float(severe_h2_shortfall_cutoff_h)
        self.severe_h2_shortfall_hours = 0.0
        self.params = HOBParams()
        self.config = HOBLoadConfig(
            maintenance_fraction_cutoff=1.0 - h2_growth_fraction_cutoff,
            min_runtime_before_replacement_h=min_runtime_before_replacement_h,
            physical_h2_feed_max_mol_h=self.physical_h2_feed_max_mol_h_value,
            enforce_h2_survival_min=True,
            auto_replace=False,
            restart_immediately_after_replacement=True,
            reset_on_shutdown=True,
        )
        self.reset()

    def reset(self):
        self.severe_h2_shortfall_hours = 0.0
        self.load = HOBHydrogenLoad(
            X0_gdw=self.fixed_X0_gdw,
            params=self.params,
            load_config=self.config,
            start_active=True,
        )

    def state(self):
        return self.load.state()

    def minimum_survival_h2_mol_h(self):
        return float(self.load.minimum_survival_h2_mol_h())

    def biological_h2_absorption_max_mol_h(self):
        return float(self.load.biological_h2_absorption_max_mol_h())

    def physical_h2_feed_max_mol_h(self):
        return float(self.load.physical_h2_feed_max_mol_h())


    def last_growth_fraction(self):
        return float(self.load.state()["last_H2_growth_fraction"])

    def age_h(self):
        return float(self.load.state()["age_h"])

    def step(
        self,
        h2_feed_mol_h,
        o2_feed_mol_h,
        co2_feed_mol_h,
        time_h,
        available_h2_mol_h=None,
        bypass_h2_feed_constraints=False,
    ):
        original_q_h2_cap = self.load.params.q_h2_hydrogenase_cap
        if bypass_h2_feed_constraints:
            self.load.params.q_h2_hydrogenase_cap = max(original_q_h2_cap, 1e12)
        try:
            row = self.load.step(
                F_H2_mol_h=float(h2_feed_mol_h),
                F_O2_mol_h=float(o2_feed_mol_h),
                F_CO2_mol_h=float(co2_feed_mol_h),
                dt_h=1.0,
                load_on=True,
                force_replace=False,
                time_h=float(time_h),
                available_H2_mol_h=None if bypass_h2_feed_constraints else available_h2_mol_h,
                enforce_H2_survival_min=False if bypass_h2_feed_constraints else True,
            )
        finally:
            self.load.params.q_h2_hydrogenase_cap = original_q_h2_cap

        physical_max = max(float(row["H2_physical_supply_max_mol_h"]), 1e-12)
        low_efficiency_replace = (
            float(row["H2_actual_feed_mol_h"]) >= 0.95 * physical_max
            and float(row["H2_growth_fraction"]) < self.h2_growth_fraction_cutoff
            and float(row["age_h"]) >= self.min_runtime_before_replacement_h
        )

        if float(row["H2_survival_shortfall_mol_h"]) > 1e-12:
            self.severe_h2_shortfall_hours += float(row["dt_h"])
        else:
            self.severe_h2_shortfall_hours = 0.0

        shortfall_hours_for_record = self.severe_h2_shortfall_hours
        severe_shortfall_replace = (
            self.severe_h2_shortfall_hours >= self.severe_h2_shortfall_cutoff_h
        )

        if low_efficiency_replace or severe_shortfall_replace:
            harvested_scp_g = self.load.replace_batch()
            row["replacement_event"] = True
            row["replacement_reason"] = (
                "severe_h2_shortfall_6h"
                if severe_shortfall_replace
                else "low_efficiency_full_feed_low_growth"
            )
            row["harvested_SCP_increment_g_protein"] = harvested_scp_g
            row["cumulative_harvested_SCP_g_protein"] = (
                self.load.cumulative_harvested_scp_g
            )
            row["X_gDW_after_event"] = self.load.X_gdw
            row["SCP_g_protein_after_event"] = self.load.SCP_g_protein
            if severe_shortfall_replace:
                self.severe_h2_shortfall_hours = 0.0

        row["low_efficiency_replacement_event"] = bool(low_efficiency_replace)
        row["severe_h2_shortfall_replacement_event"] = bool(severe_shortfall_replace)
        row["severe_h2_shortfall_hours"] = float(shortfall_hours_for_record)
        return row


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scenario_data(configs, horizon=None):
    old_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        load_data = np.genfromtxt(
            "yearly_load_data.csv",
            delimiter=",",
            skip_header=1,
            usecols=[2],
            dtype=np.float32,
        )[:horizon]
        ev_raw = np.genfromtxt(
            "UrbanEV/data/volume-11kW.csv",
            delimiter=",",
            skip_header=1,
            usecols=range(1, 8),
            dtype=np.float32,
        )
        ev_demand_full = np.nansum(ev_raw, axis=1).astype(np.float32)
        weather_path = os.path.join(BASE_DIR, "open-meteo-47.98N104.95E1092m.csv")
        t_data_C, _, w100_data_km_h, rad_data_W_m2 = get_weather_data(weather_path)
    finally:
        os.chdir(old_cwd)

    if horizon is None:
        horizon = len(ev_demand_full)

    t_data_C = np.asarray(t_data_C[:horizon], dtype=np.float32)
    w100_data_km_h = np.asarray(w100_data_km_h[:horizon], dtype=np.float32)
    rad_data_W_m2 = np.asarray(rad_data_W_m2[:horizon], dtype=np.float32)
    ev_demand = np.asarray(ev_demand_full[:horizon], dtype=np.float32)
    load_data = np.asarray(load_data[:horizon], dtype=np.float32)

    if min(len(load_data), len(ev_demand), len(t_data_C), len(w100_data_km_h), len(rad_data_W_m2)) < horizon:
        raise ValueError("The load/weather/EV data are shorter than the requested horizon.")

    pv_power = PV(configs)
    wind_power = wind_turbine(configs)
    pv_data = np.asarray(
        [pv_power(rad_data_W_m2[i], t_data_C[i]) * 1e-6 for i in range(horizon)],
        dtype=np.float32,
    )
    pv_shift_h = 6
    if horizon > pv_shift_h:
        pv_data = np.concatenate(
            [np.zeros(pv_shift_h, dtype=np.float32), pv_data[:-pv_shift_h]]
        ).astype(np.float32)
    else:
        pv_data = np.zeros_like(pv_data, dtype=np.float32)
    wind_data = np.asarray(
        [wind_power(w100_data_km_h[i]) * 10 / 36 for i in range(horizon)],
        dtype=np.float32,
    )
    return ScenarioData(
        pv=pv_data,
        wind=wind_data,
        load=load_data,
        ev_demand=ev_demand,
        t_out=t_data_C,
        wind100=w100_data_km_h,
        irradiance=rad_data_W_m2,
    )


def _hydrogen_mol_h_to_electrolyzer_mw(h2_mol_h, h2_electrolyzer_kwh_per_mol):
    h2_mol_h = np.asarray(h2_mol_h, dtype=np.float32)
    return h2_mol_h * float(h2_electrolyzer_kwh_per_mol) / 1000.0


def _electrolyzer_mw_to_hydrogen_mol_h(p_ez_mw, h2_electrolyzer_kwh_per_mol):
    p_ez_mw = np.asarray(p_ez_mw, dtype=np.float32)
    return p_ez_mw * 1000.0 / max(float(h2_electrolyzer_kwh_per_mol), 1e-12)


def match_source_to_load_ratio(data, configs, bio_h2_feed_max_mol_h, csv_path=None):
    h2_electrolyzer_kwh_per_mol = configs.h2_electrolyzer_kwh_per_mol
    source_mwh_before = float(np.sum(data.pv + data.wind))
    raw_electric_load_mwh = float(np.sum(data.load))

    horizon = len(data.load)
    bio_h2_electric_mw = float(_hydrogen_mol_h_to_electrolyzer_mw(bio_h2_feed_max_mol_h, h2_electrolyzer_kwh_per_mol))
    bio_h2_electric_mwh = float(bio_h2_electric_mw * horizon)
    bio_h2_total_mol = float(bio_h2_feed_max_mol_h * horizon)

    ev_to_bio_h2_ratio = float(getattr(configs, "ev_to_bio_h2_ratio", 0.2))
    electric_load_to_bio_electric_ratio = float(getattr(configs, "electric_load_to_bio_electric_ratio", 0.2))

    raw_ev_h2_mol_h = data.ev_demand * configs.ev2fcev_ratio_mol
    raw_ev_h2_total_mol = float(np.sum(raw_ev_h2_mol_h))
    target_ev_h2_total_mol = bio_h2_total_mol * ev_to_bio_h2_ratio
    if raw_ev_h2_total_mol <= 0.0:
        raise ValueError("Total EV hydrogen demand must be positive for EV-to-bio scaling.")
    ev_demand_scale = target_ev_h2_total_mol / raw_ev_h2_total_mol
    matched_ev_demand = (data.ev_demand * ev_demand_scale).astype(np.float32)
    ev_h2_mol_h = matched_ev_demand * configs.ev2fcev_ratio_mol
    ev_h2_electric_mw = _hydrogen_mol_h_to_electrolyzer_mw(ev_h2_mol_h, h2_electrolyzer_kwh_per_mol)
    ev_h2_electric_mwh = float(np.sum(ev_h2_electric_mw))

    target_electric_load_mwh = bio_h2_electric_mwh * electric_load_to_bio_electric_ratio
    if raw_electric_load_mwh <= 0.0:
        raise ValueError("Total electric load must be positive for electric-load-to-bio scaling.")
    electric_load_scale = target_electric_load_mwh / raw_electric_load_mwh
    matched_load = (data.load * electric_load_scale).astype(np.float32)
    electric_load_mwh = float(np.sum(matched_load))

    total_load_mwh = electric_load_mwh + ev_h2_electric_mwh + bio_h2_electric_mwh
    target_source_mwh = float(configs.source_to_load_ratio * total_load_mwh)

    if source_mwh_before <= 0.0:
        raise ValueError("Total renewable source energy must be positive for source-load matching.")

    source_scale = target_source_mwh / source_mwh_before
    matched = ScenarioData(
        pv=(data.pv * source_scale).astype(np.float32),
        wind=(data.wind * source_scale).astype(np.float32),
        load=matched_load,
        ev_demand=matched_ev_demand,
        t_out=data.t_out,
        wind100=data.wind100,
        irradiance=data.irradiance,
    )

    source_after_mw = matched.pv + matched.wind
    total_load_equivalent_mw = matched.load + ev_h2_electric_mw + bio_h2_electric_mw
    summary = {
        "source_to_load_ratio": float(configs.source_to_load_ratio),
        "source_scale": float(source_scale),
        "source_mwh_before": source_mwh_before,
        "source_mwh_after": float(np.sum(source_after_mw)),
        "raw_electric_load_mwh": raw_electric_load_mwh,
        "electric_load_mwh": electric_load_mwh,
        "electric_load_scale": float(electric_load_scale),
        "electric_load_to_bio_electric_ratio": electric_load_to_bio_electric_ratio,
        "raw_ev_h2_total_mol": raw_ev_h2_total_mol,
        "ev_h2_total_mol": float(np.sum(ev_h2_mol_h)),
        "ev_demand_scale": float(ev_demand_scale),
        "ev_to_bio_h2_ratio": ev_to_bio_h2_ratio,
        "ev_h2_electric_mwh": ev_h2_electric_mwh,
        "bio_h2_total_mol": bio_h2_total_mol,
        "bio_h2_electric_mwh": bio_h2_electric_mwh,
        "total_load_equivalent_mwh": total_load_mwh,
        "target_source_mwh": target_source_mwh,
        "bio_h2_feed_max_mol_h": float(bio_h2_feed_max_mol_h),
        "h2_electrolyzer_kwh_per_mol": float(configs.h2_electrolyzer_kwh_per_mol),
    }
    if csv_path is not None:
        folder = os.path.dirname(csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time",
                "PV(kW)",
                "Wind(kW)",
                "HFCV(mol)",
                "HFCV(kW)",
                "bioload(mol)",
                "bioload(kW)",
                "Load(kW)",
            ])
            for i in range(horizon):
                writer.writerow([
                    i,
                    float(matched.pv[i] * 1000.0),
                    float(matched.wind[i] * 1000.0),
                    float(ev_h2_mol_h[i]),
                    float(ev_h2_mol_h[i] * h2_electrolyzer_kwh_per_mol),
                    float(bio_h2_feed_max_mol_h),
                    float(bio_h2_feed_max_mol_h * h2_electrolyzer_kwh_per_mol),
                    float(matched.load[i] * 1000.0),
                ])
    return matched, summary

def split_scenario_data(data):
    total_hours = len(data.ev_demand)
    train_hours = int(total_hours * 4 / 6)
    test_hours = int(round(total_hours * 1 / 6))
    validation_hours = total_hours - train_hours - test_hours
    split_hours = {
        "train": (0, train_hours),
        "test": (train_hours, train_hours + test_hours),
        "validation": (train_hours + test_hours, total_hours),
    }

    def make_slice(start, end):
        return ScenarioData(
            pv=data.pv[start:end],
            wind=data.wind[start:end],
            load=data.load[start:end],
            ev_demand=data.ev_demand[start:end],
            t_out=data.t_out[start:end],
            wind100=data.wind100[start:end],
            irradiance=data.irradiance[start:end],
        )

    split_data = {name: make_slice(*bounds) for name, bounds in split_hours.items()}
    split_lengths = {"train": train_hours, "test": test_hours, "validation": validation_hours}
    return split_data, split_lengths


def slice_scenario_window(data, start_idx, window_length):
    start_idx = int(start_idx)
    window_length = int(window_length)
    if window_length <= 0:
        raise ValueError("window_length must be > 0")
    end_idx = start_idx + window_length
    total_hours = len(data.pv)
    if start_idx < 0 or end_idx > total_hours:
        raise ValueError(
            f"Scenario window [{start_idx}, {end_idx}) exceeds available hours {total_hours}."
        )
    return ScenarioData(
        pv=data.pv[start_idx:end_idx],
        wind=data.wind[start_idx:end_idx],
        load=data.load[start_idx:end_idx],
        ev_demand=data.ev_demand[start_idx:end_idx],
        t_out=data.t_out[start_idx:end_idx],
        wind100=data.wind100[start_idx:end_idx],
        irradiance=data.irradiance[start_idx:end_idx],
    )


def build_norm_stats(data):
    return {
        "pv_min": float(np.min(data.pv)),
        "pv_max": float(np.max(data.pv)),
        "wind_min": float(np.min(data.wind)),
        "wind_max": float(np.max(data.wind)),
        "load_min": float(np.min(data.load)),
        "load_max": float(np.max(data.load)),
        "t_out_min": float(np.min(data.t_out)),
        "t_out_max": float(np.max(data.t_out)),
    }

def plot_bio_microgrid_results(
    pv,
    wind,
    battery_discharge,
    curtail,
    load,
    ez,
    building,
    battery_charge,
    h2,
    T_room,
    T_out,
    T_wall,
    save_path,
):
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    pv = np.asarray(pv)
    wind = np.asarray(wind)
    load = np.asarray(load)
    ez = np.asarray(ez)
    curtail = np.asarray(curtail)
    building = np.asarray(building)
    battery_discharge = np.asarray(battery_discharge)
    battery_charge = np.asarray(battery_charge)
    hours = np.arange(len(pv))

    fig, ax = plt.subplots(3, 1, figsize=(30, 10))

    ax[0].bar(hours, pv, label="PV", color="#fa0505", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, wind, bottom=pv, label="Wind", color="#0095ff", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, battery_discharge, bottom=pv + wind, label="Battery discharge", color="#ffb000", edgecolor="black", linewidth=0.05)

    neg_load = -load
    neg_ez = -ez
    neg_curtail = -curtail
    neg_building = -building
    neg_battery_charge = -battery_charge
    ax[0].bar(hours, neg_load, label="Load", color="black", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, neg_ez, bottom=neg_load, label="Electrolyzer", color="#c300ff", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, neg_battery_charge, bottom=neg_load + neg_ez, label="Battery charge", color="#845EC2", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, neg_curtail, bottom=neg_load + neg_ez + neg_battery_charge, label="Curtail", color="#00ff4c", edgecolor="black", linewidth=0.05)
    ax[0].bar(hours, neg_building, bottom=neg_load + neg_ez + neg_battery_charge + neg_curtail, label="Building", color="#00D9FF9F", edgecolor="black", linewidth=0.05)
    ax[0].axhline(0, color="black", linewidth=1.5)
    ax[0].set_title("Overall Energy Management Strategy", fontsize=16)
    ax[0].set_xlabel("Time (Hour)", fontsize=12)
    ax[0].set_ylabel("Power (MW)", fontsize=12)
    ax[0].grid(True, linestyle="--", alpha=0.3)
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.1, 1), borderaxespad=0.0)

    ax[1].plot(h2, label="h2", color="blue", linewidth=2)
    ax[1].set_title("h2 tank volume")
    ax[1].set_xlabel("Time (Hour)", fontsize=12)
    ax[1].set_ylabel("mol", fontsize=12)

    ax[2].plot(T_room, label="T_room", color="blue", linewidth=2.0)
    ax[2].plot(T_out, label="T_out", color="green", linewidth=2.0)
    ax[2].plot(T_wall, label="T_wall", color="black", linewidth=2.0)
    ax[2].set_xlabel("Time (Hour)", fontsize=12)
    ax[2].set_ylabel("celcius", fontsize=12)
    ax[2].legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"结果图片已成功保存至: {save_path}")


class MicrogridBioSACEnv:
    def __init__(
        self,
        configs,
        pv_data,
        wind_data,
        load_data,
        ev_demand,
        t_data_C,
        rad_data_W_m2,
        norm_stats=None,
    ):
        self.configs = configs
        initial_data = ScenarioData(
            pv=np.asarray(pv_data, dtype=np.float32),
            wind=np.asarray(wind_data, dtype=np.float32),
            load=np.asarray(load_data, dtype=np.float32),
            ev_demand=np.asarray(ev_demand, dtype=np.float32),
            t_out=np.asarray(t_data_C, dtype=np.float32),
            wind100=np.zeros_like(np.asarray(pv_data, dtype=np.float32)),
            irradiance=np.asarray(rad_data_W_m2, dtype=np.float32),
        )
        self.norm_stats = dict(norm_stats) if norm_stats is not None else build_norm_stats(initial_data)
        self.set_scenario_data(initial_data)

        self.building = smart_building()
        self.bio_load = FixedStrainHydrogenLoad(
            fixed_X0_gdw=100.0,
            h2_growth_fraction_cutoff=0.5,
            min_runtime_before_replacement_h=24.0,
            physical_h2_feed_max_mol_h=5000.0,
        )
        self.h2_electrolyzer_kwh_per_mol = configs.h2_electrolyzer_kwh_per_mol

        self.h2_min = configs.h2_tank_mol_min * configs.h2_tank_num
        self.h2_max = configs.h2_tank_mol_max * configs.h2_tank_num
        self.h2_initial = (self.h2_min + self.h2_max) * 0.5

        self.p_ez_max_mw = 50.0
        self.p_hvac_max_kw = 100.0
        self.bio_h2_feed_max_mol_h = self.bio_load.physical_h2_feed_max_mol_h()
        self.bio_o2_ratio = 0.40
        self.bio_co2_ratio = 0.125

        self.dispatch_time_step_h = 1.0
        self.reward_scale_yuan = 1000.0
        self.curtail_price_yuan_per_kwh = 1.0
        self.grid_price_yuan_per_kwh = 1.0
        self.carbon_price_yuan_per_tco2 = 577.0
        self.scp_price_yuan_per_kg_protein = 12.0
        self.h2_violation_penalty_yuan = 1000.0
        self.battery_violation_penalty_yuan = 1000.0
        self.comfort_violation_penalty_yuan = 1000.0
        self.bio_forced_replacement_penalty_yuan = 2000.0
        self.bio_h2_violation_penalty_yuan = 1000.0
        self.battery_capacity_mwh = 2.0
        self.battery_power_max_mw = 2.0
        self.battery_efficiency = 0.9
        self.battery_initial_mwh = 0.5 * self.battery_capacity_mwh

        self.state_dim = 18
        self.action_dim = 4
        self.reset()

    def set_scenario_data(self, scenario_data):
        self.pv = np.asarray(scenario_data.pv, dtype=np.float32)
        self.wind = np.asarray(scenario_data.wind, dtype=np.float32)
        self.load = np.asarray(scenario_data.load, dtype=np.float32)
        self.ev_demand = np.asarray(scenario_data.ev_demand, dtype=np.float32)
        self.ev_h2_out_mol_h = self.ev_demand * self.configs.ev2fcev_ratio_mol
        self.ev_h2_out_max_mol_h = max(float(np.max(self.ev_h2_out_mol_h)), 1.0)
        self.t_out = np.asarray(scenario_data.t_out, dtype=np.float32)
        self.rad = np.asarray(scenario_data.irradiance, dtype=np.float32)
        self.horizon = len(self.pv)

    def reset(self):
        self.t = 0
        self.h2_tank_mol = self.h2_initial
        self.T_in = 24.0
        self.T_wall = 15.0
        self.battery_soc_mwh = self.battery_initial_mwh
        self.bio_load.reset()
        self.records = []
        return self._state()

    def _safe_scale(self, value, key):
        low = float(self.norm_stats[f"{key}_min"])
        high = float(self.norm_stats[f"{key}_max"])
        return (float(value) - low) / (high - low + 1e-6)

    def _available_power_for_h2_mw(self, idx, building_mw=0.0):
        source_mw = float(self.pv[idx] + self.wind[idx])
        electric_load_mw = float(self.load[idx] + building_mw)
        return max(0.0, source_mw - electric_load_mw)

    def _apply_battery_command(self, command_mw, charge_power_limit_mw=None):
        requested_command_mw = float(command_mw)
        command_mw = float(np.clip(requested_command_mw, -self.battery_power_max_mw, self.battery_power_max_mw))
        soc_before = self.battery_soc_mwh
        charge_mw = 0.0
        discharge_mw = 0.0

        battery_power_limit_violation_mw = max(
            0.0, abs(requested_command_mw) - self.battery_power_max_mw
        )
        if command_mw >= 0.0:
            requested_soc_after = soc_before - command_mw / self.battery_efficiency
        else:
            requested_soc_after = soc_before + (-command_mw) * self.battery_efficiency
        requested_soc_fraction_after = requested_soc_after / max(self.battery_capacity_mwh, 1e-12)
        battery_soc_lower_violation_mwh = max(0.0, -requested_soc_after)
        battery_soc_upper_violation_mwh = max(0.0, requested_soc_after - self.battery_capacity_mwh)
        battery_soc_violation_mwh = (
            battery_soc_lower_violation_mwh + battery_soc_upper_violation_mwh
        )
        battery_soc_lower_violation_event = battery_soc_lower_violation_mwh > 1e-9
        battery_soc_upper_violation_event = battery_soc_upper_violation_mwh > 1e-9
        battery_soc_violation_event = (
            battery_soc_lower_violation_event or battery_soc_upper_violation_event
        )
        battery_violation_event = battery_soc_violation_event

        if command_mw >= 0.0:
            discharge_mw = command_mw
            self.battery_soc_mwh -= discharge_mw / self.battery_efficiency
        else:
            charge_mw = -command_mw
            self.battery_soc_mwh += charge_mw * self.battery_efficiency

        self.battery_soc_mwh = float(self.battery_soc_mwh)
        return {
            "battery_requested_command_mw": requested_command_mw,
            "battery_command_mw": command_mw,
            "battery_charge_mw": float(charge_mw),
            "battery_discharge_mw": float(discharge_mw),
            "battery_power_mw": float(discharge_mw - charge_mw),
            "battery_soc_before_mwh": float(soc_before),
            "battery_soc_mwh": float(self.battery_soc_mwh),
            "battery_soc_fraction": float(self.battery_soc_mwh / max(self.battery_capacity_mwh, 1e-6)),
            "battery_power_limit_violation_mw": float(battery_power_limit_violation_mw),
            "battery_requested_soc_fraction_after": float(requested_soc_fraction_after),
            "battery_soc_lower_violation_mwh": float(battery_soc_lower_violation_mwh),
            "battery_soc_upper_violation_mwh": float(battery_soc_upper_violation_mwh),
            "battery_soc_violation_mwh": float(battery_soc_violation_mwh),
            "battery_soc_lower_violation_event": float(battery_soc_lower_violation_event),
            "battery_soc_upper_violation_event": float(battery_soc_upper_violation_event),
            "battery_soc_violation_event": float(battery_soc_violation_event),
            "battery_violation_event": float(battery_violation_event),
        }

    def _state(self):
        idx = min(self.t, self.horizon - 1)
        bio_state = self.bio_load.state()
        h2_norm = (self.h2_tank_mol - self.h2_min) / (self.h2_max - self.h2_min + 1e-6)
        min_survival_h2 = self.bio_load.minimum_survival_h2_mol_h()
        growth_fraction = self.bio_load.last_growth_fraction()
        bio_age_h = self.bio_load.age_h()
        available_h2_power_mw = self._available_power_for_h2_mw(idx)
        battery_soc_fraction = self.battery_soc_mwh / max(self.battery_capacity_mwh, 1e-6)
        ev_h2_out_fraction = self.ev_h2_out_mol_h[idx] / max(self.ev_h2_out_max_mol_h, 1e-6)
        return np.asarray(
            [
                idx / max(self.horizon - 1, 1),
                np.sin(2 * np.pi * (idx % 24) / 24),
                np.cos(2 * np.pi * (idx % 24) / 24),
                h2_norm,
                battery_soc_fraction,
                (self.T_in + 20.0) / 60.0,
                (self.T_wall + 20.0) / 60.0,
                np.log1p(bio_state["X_gDW"]) / 12.0,
                np.log1p(bio_state["SCP_g_protein"]) / 12.0,
                min_survival_h2 / max(self.bio_h2_feed_max_mol_h, 1e-6),
                growth_fraction,
                bio_age_h / max(self.bio_load.config.min_runtime_before_replacement_h, 1.0),
                available_h2_power_mw / max(self.p_ez_max_mw, 1e-6),
                self._safe_scale(self.pv[idx], "pv"),
                self._safe_scale(self.wind[idx], "wind"),
                self._safe_scale(self.load[idx], "load"),
                self._safe_scale(self.t_out[idx], "t_out"),
                ev_h2_out_fraction,
            ],
            dtype=np.float32,
        )

    def _map_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled = (action + 1.0) * 0.5
        p_ez_mw = float(scaled[0] * self.p_ez_max_mw)
        bio_h2_feed_mol_h = float(scaled[1] * self.bio_h2_feed_max_mol_h)
        p_hvac_request_kw = float(scaled[2] * self.p_hvac_max_kw)
        battery_command_mw = float(action[3] * self.battery_power_max_mw)
        return p_ez_mw, bio_h2_feed_mol_h, p_hvac_request_kw, battery_command_mw

    def _physical_to_action(self, p_ez_mw, bio_h2_feed_mol_h, p_hvac_kw, battery_power_mw):
        return np.asarray(
            [
                np.clip(2.0 * p_ez_mw / max(self.p_ez_max_mw, 1e-6) - 1.0, -1.0, 1.0),
                np.clip(2.0 * bio_h2_feed_mol_h / max(self.bio_h2_feed_max_mol_h, 1e-6) - 1.0, -1.0, 1.0),
                np.clip(2.0 * p_hvac_kw / max(self.p_hvac_max_kw, 1e-6) - 1.0, -1.0, 1.0),
                np.clip(battery_power_mw / max(self.battery_power_max_mw, 1e-6), -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _bio_step(self, h2_feed_mol_h, available_h2_mol_h=None):
        row = self.bio_load.step(
            h2_feed_mol_h=h2_feed_mol_h,
            o2_feed_mol_h=h2_feed_mol_h * self.bio_o2_ratio,
            co2_feed_mol_h=h2_feed_mol_h * self.bio_co2_ratio,
            time_h=self.t + 1,
            available_h2_mol_h=available_h2_mol_h,
            bypass_h2_feed_constraints=True,
        )
        return {
            "bio_h2_feed_mol_h": float(row["H2_input_mol_h"]),
            "bio_h2_requested_mol_h": float(row["H2_requested_mol_h"]),
            "bio_h2_actual_feed_mol_h": float(row["H2_actual_feed_mol_h"]),
            "bio_h2_survival_min_mol_h": float(row["H2_survival_min_mol_h"]),
            "bio_h2_biological_absorption_max_mol_h": float(row["H2_biological_absorption_max_mol_h"]),
            "bio_h2_physical_supply_max_mol_h": float(row["H2_physical_supply_max_mol_h"]),
            "bio_h2_external_available_mol_h": float(row["H2_external_available_mol_h"]),
            "bio_h2_feasible_min_mol_h": float(row["H2_feasible_min_mol_h"]),
            "bio_h2_feasible_max_mol_h": float(row["H2_feasible_max_mol_h"]),
            "bio_h2_feed_raised_to_survival": float(row["H2_feed_raised_to_survival"]),
            "bio_h2_feed_limited_by_upper": float(row["H2_feed_limited_by_upper"]),
            "bio_h2_survival_shortfall_mol_h": float(row["H2_survival_shortfall_mol_h"]),
            "bio_o2_feed_mol_h": float(row["O2_input_mol_h"]),
            "bio_co2_feed_mol_h": float(row["CO2_input_mol_h"]),
            "bio_h2_load_mol_h": float(row["H2_load_mol_h"]),
            "bio_h2_uptake_mol": float(row["H2_uptake_mol"]),
            "bio_o2_uptake_mol": float(row["O2_uptake_mol"]),
            "bio_co2_uptake_mol": float(row["CO2_uptake_mol"]),
            "bio_h2_unused_mol": float(row["H2_unused_mol"]),
            "bio_X_gDW": float(row["X_gDW"]),
            "bio_X_gDW_after_event": float(row["X_gDW_after_event"]),
            "bio_SCP_g_protein": float(row["SCP_g_protein"]),
            "bio_SCP_g_protein_after_event": float(row["SCP_g_protein_after_event"]),
            "bio_dX_gDW": float(row["dX_gDW"]),
            "bio_dSCP_g_protein": float(row["dSCP_g_protein"]),
            "bio_harvested_SCP_g_protein": float(row["harvested_SCP_increment_g_protein"]),
            "bio_cumulative_harvested_SCP_g_protein": float(row["cumulative_harvested_SCP_g_protein"]),
            "bio_mu_h_1": float(row["mu_h-1"]),
            "bio_starvation_level": float(row["starvation_level"]),
            "bio_H2_maintenance_fraction": float(row["H2_maintenance_fraction"]),
            "bio_H2_growth_fraction": float(row["H2_growth_fraction"]),
            "bio_replacement_event": float(row["replacement_event"]),
            "bio_replacement_reason": str(row.get("replacement_reason", "none")),
            "bio_low_efficiency_replacement_event": float(row.get("low_efficiency_replacement_event", False)),
            "bio_severe_h2_shortfall_replacement_event": float(row.get("severe_h2_shortfall_replacement_event", False)),
            "bio_severe_h2_shortfall_hours": float(row.get("severe_h2_shortfall_hours", 0.0)),
            "bio_cycle_id": float(row["cycle_id"]),
            "bio_age_h": float(row["age_h"]),
        }

    def step(self, action):
        idx = self.t
        p_ez_command_mw, bio_h2_feed_mol_h, p_hvac_request_kw, battery_command_mw = self._map_action(action)
        p_hvac_electric_kw = float(np.clip(p_hvac_request_kw, 0.0, self.p_hvac_max_kw))
        if self.T_in < self.building.comfort_temp_min:
            hvac_mode = "heating"
            p_heat_kw = p_hvac_electric_kw
            p_cool_kw = 0.0
        elif self.T_in > self.building.pre_cooling_temp:
            hvac_mode = "cooling"
            p_heat_kw = 0.0
            p_cool_kw = p_hvac_electric_kw
        else:
            hvac_mode = "on_idle"
            p_heat_kw = 0.0
            p_cool_kw = 0.0
        building_mw = p_hvac_electric_kw * 1e-3 * self.configs.building_num

        supply = float(self.pv[idx] + self.wind[idx])
        electric_load_mw = float(self.load[idx] + building_mw)
        battery = self._apply_battery_command(battery_command_mw)
        supply_with_battery_mw = supply + battery["battery_discharge_mw"]
        available_h2_power_mw = max(
            0.0,
            supply_with_battery_mw - electric_load_mw - battery["battery_charge_mw"],
        )
        p_ez_mw = float(np.clip(p_ez_command_mw, 0.0, self.p_ez_max_mw))
        p_ez_clipped_mw = max(0.0, p_ez_command_mw - p_ez_mw)
        demand = electric_load_mw + battery["battery_charge_mw"] + p_ez_mw
        surplus = supply_with_battery_mw - demand
        curtail_mw = max(surplus, 0.0)
        grid_purchase_mwh = max(-surplus, 0.0) * self.dispatch_time_step_h

        bio_min_survival_h2_mol_h = self.bio_load.minimum_survival_h2_mol_h()
        bio_max_demand_h2_mol_h = self.bio_load.biological_h2_absorption_max_mol_h()
        h2_in_mol_h = p_ez_mw * 1000.0 / max(self.h2_electrolyzer_kwh_per_mol, 1e-12)
        ev_h2_out_mol_h = float(self.ev_h2_out_mol_h[idx])
        h2_after_ev_mol = self.h2_tank_mol - ev_h2_out_mol_h
        bio_available_h2_mol_h = max(0.0, h2_after_ev_mol + h2_in_mol_h - self.h2_min)
        bio_h2_feed_requested_mol_h = bio_h2_feed_mol_h
        bio_h2_feed_mol_h = float(
            np.clip(bio_h2_feed_requested_mol_h, 0.0, self.bio_h2_feed_max_mol_h)
        )
        bio = self._bio_step(bio_h2_feed_mol_h, available_h2_mol_h=None)
        next_h2 = h2_after_ev_mol + h2_in_mol_h - bio["bio_h2_uptake_mol"]
        h2_lower_violation_mol = max(self.h2_min - next_h2, 0.0)
        h2_upper_violation_mol = max(next_h2 - self.h2_max, 0.0)
        h2_violation = h2_lower_violation_mol + h2_upper_violation_mol
        self.h2_tank_mol = float(next_h2)
        executed_action = self._physical_to_action(
            p_ez_mw,
            bio["bio_h2_feed_mol_h"],
            p_hvac_electric_kw,
            battery["battery_power_mw"],
        )

        q_from_wall = (self.T_wall - self.T_in) / self.building.r1
        q_from_out = (float(self.t_out[idx]) - self.T_in) / self.building.rwind
        q_hvac = self.building.COP_heating * p_heat_kw - self.building.COP_cooling * p_cool_kw
        next_T_in = self.T_in + (q_from_wall + q_from_out + q_hvac) / self.building.czone

        q_out_to_wall = (float(self.t_out[idx]) - self.T_wall) / self.building.r1
        q_in_to_wall = (self.T_in - self.T_wall) / self.building.r2
        irradiance_proxy = self.pv[idx] * 1000 / 0.8 / 100
        q_solar = self.building.Gi_solar * irradiance_proxy
        next_T_wall = self.T_wall + (q_out_to_wall + q_in_to_wall + q_solar) / self.building.c

        self.T_in = float(np.clip(next_T_in, -20.0, 40.0))
        self.T_wall = float(np.clip(next_T_wall, -20.0, 40.0))

        comfort_violation = self.building.comfort_violation(self.T_in)
        comfort_cost = comfort_violation**2
        comfort_violation_event = bool(comfort_violation > 1e-9)

        curtail_mwh = curtail_mw * self.dispatch_time_step_h
        curtail_cost_yuan = (
            curtail_mwh * 1000.0 * self.curtail_price_yuan_per_kwh
        )
        grid_purchase_cost_yuan = (
            grid_purchase_mwh * 1000.0 * self.grid_price_yuan_per_kwh
        )

        h2_lower_violation_event = bool(h2_lower_violation_mol > 1e-9)
        h2_upper_violation_event = bool(h2_upper_violation_mol > 1e-9)
        h2_violation_event = h2_lower_violation_event or h2_upper_violation_event
        h2_lower_violation_cost_yuan = (
            self.h2_violation_penalty_yuan if h2_lower_violation_event else 0.0
        )
        h2_upper_violation_cost_yuan = (
            self.h2_violation_penalty_yuan if h2_upper_violation_event else 0.0
        )
        h2_violation_cost_yuan = (
            h2_lower_violation_cost_yuan + h2_upper_violation_cost_yuan
        )
        comfort_violation_cost_yuan = (
            self.comfort_violation_penalty_yuan if comfort_violation_event else 0.0
        )
        battery_soc_lower_violation_event = bool(
            battery["battery_soc_lower_violation_event"] > 0.5
        )
        battery_soc_upper_violation_event = bool(
            battery["battery_soc_upper_violation_event"] > 0.5
        )
        battery_soc_violation_event = (
            battery_soc_lower_violation_event or battery_soc_upper_violation_event
        )
        battery_soc_lower_violation_cost_yuan = (
            self.battery_violation_penalty_yuan
            if battery_soc_lower_violation_event
            else 0.0
        )
        battery_soc_upper_violation_cost_yuan = (
            self.battery_violation_penalty_yuan
            if battery_soc_upper_violation_event
            else 0.0
        )
        battery_violation_event = battery_soc_violation_event
        battery_violation_cost_yuan = (
            battery_soc_lower_violation_cost_yuan + battery_soc_upper_violation_cost_yuan
        )

        grid_co2_emission_t = grid_purchase_mwh * self.configs.grid_emission_factor_tco2_per_mwh
        bio_co2_absorption_t = bio["bio_co2_uptake_mol"] * self.configs.co2_molar_mass_g_per_mol / 1e6
        net_co2_emission_t = grid_co2_emission_t - bio_co2_absorption_t
        carbon_cost_yuan = max(net_co2_emission_t, 0.0) * self.carbon_price_yuan_per_tco2
        carbon_credit_yuan = max(-net_co2_emission_t, 0.0) * self.carbon_price_yuan_per_tco2

        scp_growth_revenue_yuan = (
            max(bio["bio_dSCP_g_protein"], 0.0) / 1000.0 * self.scp_price_yuan_per_kg_protein
        )
        scp_harvest_revenue_yuan = (
            max(bio["bio_harvested_SCP_g_protein"], 0.0) / 1000.0 * self.scp_price_yuan_per_kg_protein
        )

        bio_forced_replacement_event = bool(
            bio["bio_severe_h2_shortfall_replacement_event"]
        )
        bio_forced_replacement_cost_yuan = (
            self.bio_forced_replacement_penalty_yuan
            if bio_forced_replacement_event
            else 0.0
        )
        bio_starvation_shortage_mol_h = max(
            0.0, bio_min_survival_h2_mol_h - bio["bio_h2_load_mol_h"]
        )
        bio_starvation_cost = 0.0
        bio_h2_lower_demand_violation_mol_h = max(
            0.0, bio_min_survival_h2_mol_h - bio["bio_h2_feed_mol_h"]
        )
        bio_h2_upper_demand_violation_mol_h = max(
            0.0, bio["bio_h2_feed_mol_h"] - bio_max_demand_h2_mol_h
        )
        bio_h2_lower_demand_violation_event = bool(
            bio_h2_lower_demand_violation_mol_h > 1e-9
        )
        bio_h2_upper_demand_violation_event = bool(
            bio_h2_upper_demand_violation_mol_h > 1e-9
        )
        bio_h2_violation_event = (
            bio_h2_lower_demand_violation_event
            or bio_h2_upper_demand_violation_event
        )
        bio_h2_lower_demand_violation_cost_yuan = (
            self.bio_h2_violation_penalty_yuan
            if bio_h2_lower_demand_violation_event
            else 0.0
        )
        bio_h2_upper_demand_violation_cost_yuan = (
            self.bio_h2_violation_penalty_yuan
            if bio_h2_upper_demand_violation_event
            else 0.0
        )
        bio_h2_violation_cost_yuan = (
            bio_h2_lower_demand_violation_cost_yuan
            + bio_h2_upper_demand_violation_cost_yuan
        )

        total_revenue_yuan = (
            scp_growth_revenue_yuan
            + scp_harvest_revenue_yuan
            + carbon_credit_yuan
        )
        total_cost_yuan = (
            curtail_cost_yuan
            + grid_purchase_cost_yuan
            + h2_violation_cost_yuan
            + battery_violation_cost_yuan
            + comfort_violation_cost_yuan
            + carbon_cost_yuan
            + bio_forced_replacement_cost_yuan
            + bio_h2_violation_cost_yuan
        )
        step_profit_yuan = total_revenue_yuan - total_cost_yuan
        step_cost = total_cost_yuan - total_revenue_yuan
        reward = step_profit_yuan / max(self.reward_scale_yuan, 1e-12)

        weighted_comfort_cost = comfort_violation_cost_yuan
        h2_cost = h2_violation_cost_yuan
        carbon_cost = carbon_cost_yuan
        biomass_reward = scp_growth_revenue_yuan
        harvest_reward = scp_harvest_revenue_yuan
        bio_forced_replacement_cost = bio_forced_replacement_cost_yuan

        self.records.append(
            {
                "source_mw": supply,
                "source_with_battery_mw": supply_with_battery_mw,
                "electric_load_mw": electric_load_mw,
                "available_h2_power_mw": available_h2_power_mw,
                "p_ez_command_mw": p_ez_command_mw,
                "p_ez_mw": p_ez_mw,
                "p_ez_clipped_mw": p_ez_clipped_mw,
                **battery,
                "p_curtail_mw": curtail_mw,
                "curtail_mwh": curtail_mwh,
                "curtail_cost_yuan": curtail_cost_yuan,
                "grid_purchase_mwh": grid_purchase_mwh,
                "grid_purchase_cost_yuan": grid_purchase_cost_yuan,
                "battery_soc_lower_violation_event": float(battery_soc_lower_violation_event),
                "battery_soc_upper_violation_event": float(battery_soc_upper_violation_event),
                "battery_soc_violation_event": float(battery_soc_violation_event),
                "battery_soc_lower_violation_cost_yuan": battery_soc_lower_violation_cost_yuan,
                "battery_soc_upper_violation_cost_yuan": battery_soc_upper_violation_cost_yuan,
                "battery_violation_event": float(battery_violation_event),
                "battery_violation_cost_yuan": battery_violation_cost_yuan,
                "grid_co2_emission_t": grid_co2_emission_t,
                "bio_co2_absorption_t": bio_co2_absorption_t,
                "net_co2_emission_t": net_co2_emission_t,
                "carbon_cost": carbon_cost,
                "carbon_cost_yuan": carbon_cost_yuan,
                "carbon_credit_yuan": carbon_credit_yuan,
                "scp_growth_revenue_yuan": scp_growth_revenue_yuan,
                "scp_harvest_revenue_yuan": scp_harvest_revenue_yuan,
                "total_revenue_yuan": total_revenue_yuan,
                "total_cost_yuan": total_cost_yuan,
                "step_profit_yuan": step_profit_yuan,
                "reward_scale_yuan": self.reward_scale_yuan,
                "h2_tank_mol": self.h2_tank_mol,
                "building_mw": building_mw,
                "hvac_mode": hvac_mode,
                "p_hvac_request_kw": p_hvac_request_kw,
                "p_heat_kw": p_heat_kw,
                "p_cool_kw": p_cool_kw,
                "p_hvac_electric_kw": p_hvac_electric_kw,
                "q_hvac": q_hvac,
                "bio_min_survival_h2_mol_h": bio_min_survival_h2_mol_h,
                "bio_max_demand_h2_mol_h": bio_max_demand_h2_mol_h,
                "bio_starvation_shortage_mol_h": bio_starvation_shortage_mol_h,
                "bio_starvation_cost": bio_starvation_cost,
                "bio_h2_lower_demand_violation_mol_h": bio_h2_lower_demand_violation_mol_h,
                "bio_h2_upper_demand_violation_mol_h": bio_h2_upper_demand_violation_mol_h,
                "bio_h2_lower_demand_violation_event": float(bio_h2_lower_demand_violation_event),
                "bio_h2_upper_demand_violation_event": float(bio_h2_upper_demand_violation_event),
                "bio_h2_violation_event": float(bio_h2_violation_event),
                "bio_h2_lower_demand_violation_cost_yuan": bio_h2_lower_demand_violation_cost_yuan,
                "bio_h2_upper_demand_violation_cost_yuan": bio_h2_upper_demand_violation_cost_yuan,
                "bio_h2_violation_cost_yuan": bio_h2_violation_cost_yuan,
                "bio_forced_replacement_cost": bio_forced_replacement_cost,
                "ev_h2_out_mol_h": ev_h2_out_mol_h,
                "h2_after_ev_mol": h2_after_ev_mol,
                "bio_available_h2_mol_h": bio_available_h2_mol_h,
                "bio_h2_feed_requested_mol_h": bio_h2_feed_requested_mol_h,
                "bio_h2_feed_executed_mol_h": bio["bio_h2_feed_mol_h"],
                "bio_h2_survival_min_mol_h": bio["bio_h2_survival_min_mol_h"],
                "bio_h2_biological_absorption_max_mol_h": bio["bio_h2_biological_absorption_max_mol_h"],
                "bio_h2_physical_supply_max_mol_h": bio["bio_h2_physical_supply_max_mol_h"],
                "bio_h2_feasible_min_mol_h": bio["bio_h2_feasible_min_mol_h"],
                "bio_h2_feasible_max_mol_h": bio["bio_h2_feasible_max_mol_h"],
                "bio_h2_survival_shortfall_mol_h": bio["bio_h2_survival_shortfall_mol_h"],
                "executed_action": executed_action,
                **bio,
                "T_room": self.T_in,
                "T_wall": self.T_wall,
                "curtail_cost": curtail_cost_yuan,
                "comfort_violation_C": comfort_violation,
                "comfort_violation_event": float(comfort_violation_event),
                "comfort_violation_cost_yuan": comfort_violation_cost_yuan,
                "comfort_cost_raw": comfort_cost,
                "comfort_cost_weighted": weighted_comfort_cost,
                "h2_tank_violation_mol": h2_violation,
                "h2_lower_violation_mol": h2_lower_violation_mol,
                "h2_upper_violation_mol": h2_upper_violation_mol,
                "h2_lower_violation_event": float(h2_lower_violation_event),
                "h2_upper_violation_event": float(h2_upper_violation_event),
                "h2_violation_event": float(h2_violation_event),
                "h2_lower_violation_cost_yuan": h2_lower_violation_cost_yuan,
                "h2_upper_violation_cost_yuan": h2_upper_violation_cost_yuan,
                "h2_violation_cost_yuan": h2_violation_cost_yuan,
                "h2_tank_violation_cost": h2_cost,
                "bio_forced_replacement_event": float(bio_forced_replacement_event),
                "bio_forced_replacement_cost_yuan": bio_forced_replacement_cost_yuan,
                "biomass_reward_component": biomass_reward,
                "harvest_reward_component": harvest_reward,
                "step_cost": step_cost,
                "reward": reward,
            }
        )

        self.t += 1
        done = self.t >= self.horizon
        next_state = self._state()
        return next_state, float(reward), done, self.records[-1]


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_states[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )



class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.trunk(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = self._make_q_network(state_dim, action_dim)
        self.q2 = self._make_q_network(state_dim, action_dim)

    @staticmethod
    def _make_q_network(state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)


class SACAgent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.cfg = cfg
        self.device = device
        self.actor = GaussianPolicy(
            state_dim,
            action_dim,
            cfg.hidden_dim,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(device)
        self.critic = TwinQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(cfg.target_entropy)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, noise_std=0.0):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if noise_std > 0:
                action, _ = self.actor.sample(state_tensor)
            else:
                action = self.actor.deterministic(state_tensor)
            action = torch.clamp(action, -1.0, 1.0)
        return np.clip(action.cpu().numpy()[0], -1.0, 1.0).astype(np.float32)

    def learn(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions, log_prob = self.actor.sample(states)
        actor_actions = torch.clamp(actor_actions, -1.0, 1.0)
        q1_pi, q2_pi = self.critic(states, actor_actions)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic_target, self.critic)
        return float(actor_loss.item()), float(critic_loss.item())

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.cfg.tau * source_param.data + (1.0 - self.cfg.tau) * target_param.data
            )


def evaluate_policy(env, agent):
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, noise_std=0.0)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, list(env.records)


def records_to_arrays(records):
    result = {}
    for key in records[0]:
        values = [row[key] for row in records]
        try:
            result[key] = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            result[key] = np.asarray(values, dtype=str)
    return result


def save_reward_components_csv(save_dir, split_name, records):
    reward_keys = [
        "hour",
        "reward",
        "step_profit_yuan",
        "step_cost",
        "total_revenue_yuan",
        "total_cost_yuan",
        "curtail_cost_yuan",
        "grid_purchase_cost_yuan",
        "h2_violation_event",
        "h2_lower_violation_event",
        "h2_upper_violation_event",
        "h2_lower_violation_cost_yuan",
        "h2_upper_violation_cost_yuan",
        "h2_violation_cost_yuan",
        "battery_violation_event",
        "battery_soc_lower_violation_event",
        "battery_soc_upper_violation_event",
        "battery_soc_violation_event",
        "battery_soc_lower_violation_cost_yuan",
        "battery_soc_upper_violation_cost_yuan",
        "battery_violation_cost_yuan",
        "comfort_violation_C",
        "comfort_violation_event",
        "comfort_violation_cost_yuan",
        "grid_co2_emission_t",
        "bio_co2_absorption_t",
        "net_co2_emission_t",
        "carbon_cost_yuan",
        "carbon_credit_yuan",
        "scp_growth_revenue_yuan",
        "scp_harvest_revenue_yuan",
        "bio_forced_replacement_event",
        "bio_forced_replacement_cost_yuan",
        "bio_h2_violation_event",
        "bio_h2_lower_demand_violation_event",
        "bio_h2_upper_demand_violation_event",
        "bio_h2_lower_demand_violation_cost_yuan",
        "bio_h2_upper_demand_violation_cost_yuan",
        "bio_h2_violation_cost_yuan",
        "grid_purchase_kwh",
        "curtail_kwh",
        "h2_tank_violation_mol",
        "h2_lower_violation_mol",
        "h2_upper_violation_mol",
        "bio_starvation_cost",
        "bio_h2_lower_demand_violation_mol_h",
        "bio_h2_upper_demand_violation_mol_h",
        "battery_charge_kw",
        "battery_discharge_kw",
        "battery_soc_kwh",
        "battery_soc_fraction",
        "biomass_reward_component",
        "harvest_reward_component",
    ]
    path = os.path.join(save_dir, f"reward_components_{split_name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reward_keys)
        writer.writeheader()
        for hour, row in enumerate(records):
            out = {key: row.get(key, 0.0) for key in reward_keys}
            out["hour"] = hour
            out["grid_purchase_kwh"] = row.get("grid_purchase_mwh", 0.0) * 1000.0
            out["curtail_kwh"] = row.get("curtail_mwh", row.get("p_curtail_mw", 0.0)) * 1000.0
            out["battery_charge_kw"] = row.get("battery_charge_mw", 0.0) * 1000.0
            out["battery_discharge_kw"] = row.get("battery_discharge_mw", 0.0) * 1000.0
            out["battery_soc_kwh"] = row.get("battery_soc_mwh", 0.0) * 1000.0
            writer.writerow(out)
    return path


def save_test_timeseries_details_csv(save_dir, records):
    fields = [
        "time_step",
        "source_kw",
        "source_with_battery_kw",
        "electric_load_kw",
        "building_hvac_electric_kw",
        "hvac_mode",
        "p_hvac_request_kw",
        "p_heat_kw",
        "p_cool_kw",
        "p_hvac_electric_kw",
        "q_hvac",
        "available_h2_power_kw",
        "p_ez_kw",
        "grid_purchase_kwh",
        "p_curtail_kw",
        "curtail_cost_yuan",
        "grid_purchase_cost_yuan",
        "h2_violation_event",
        "h2_lower_violation_event",
        "h2_upper_violation_event",
        "h2_lower_violation_cost_yuan",
        "h2_upper_violation_cost_yuan",
        "h2_violation_cost_yuan",
        "battery_violation_event",
        "battery_soc_lower_violation_event",
        "battery_soc_upper_violation_event",
        "battery_soc_violation_event",
        "battery_soc_lower_violation_cost_yuan",
        "battery_soc_upper_violation_cost_yuan",
        "battery_violation_cost_yuan",
        "comfort_violation_event",
        "comfort_violation_cost_yuan",
        "carbon_cost_yuan",
        "carbon_credit_yuan",
        "scp_growth_revenue_yuan",
        "scp_harvest_revenue_yuan",
        "bio_forced_replacement_event",
        "bio_forced_replacement_cost_yuan",
        "bio_h2_violation_event",
        "bio_h2_lower_demand_violation_event",
        "bio_h2_upper_demand_violation_event",
        "bio_h2_lower_demand_violation_cost_yuan",
        "bio_h2_upper_demand_violation_cost_yuan",
        "bio_h2_violation_cost_yuan",
        "total_revenue_yuan",
        "total_cost_yuan",
        "step_profit_yuan",
        "co2_absorbed_mol",
        "co2_absorbed_t",
        "grid_co2_emission_t",
        "net_co2_emission_t",
        "T_room_C",
        "T_wall_C",
        "scp_increment_g_protein",
        "scp_active_g_protein",
        "scp_harvested_increment_g_protein",
        "scp_cumulative_harvested_g_protein",
        "scp_total_available_g_protein",
        "bio_X_gDW",
        "bio_H2_growth_fraction",
        "bio_H2_maintenance_fraction",
        "bio_replacement_event",
        "bio_replacement_reason",
        "bio_low_efficiency_replacement_event",
        "bio_severe_h2_shortfall_replacement_event",
        "bio_severe_h2_shortfall_hours",
        "battery_charge_kw",
        "battery_discharge_kw",
        "battery_power_kw",
        "battery_soc_kwh",
        "battery_soc_fraction",
        "h2_tank_mol",
        "h2_tank_violation_mol",
        "h2_lower_violation_mol",
        "h2_upper_violation_mol",
        "ev_h2_out_mol_h",
        "h2_after_ev_mol",
        "bio_available_h2_mol_h",
        "bio_h2_feed_requested_mol_h",
        "bio_h2_feed_executed_mol_h",
        "bio_h2_actual_feed_mol_h",
        "bio_h2_load_mol_h",
        "bio_h2_uptake_mol",
        "bio_h2_survival_min_mol_h",
        "bio_max_demand_h2_mol_h",
        "bio_h2_survival_shortfall_mol_h",
        "bio_h2_biological_absorption_max_mol_h",
        "bio_h2_lower_demand_violation_mol_h",
        "bio_h2_upper_demand_violation_mol_h",
        "bio_h2_physical_supply_max_mol_h",
        "reward",
        "step_cost",
    ]
    path = os.path.join(save_dir, "test_timeseries_details.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for time_step, row in enumerate(records):
            scp_active = float(row.get("bio_SCP_g_protein_after_event", row.get("bio_SCP_g_protein", 0.0)))
            scp_harvested_total = float(row.get("bio_cumulative_harvested_SCP_g_protein", 0.0))
            out = {
                "time_step": time_step,
                "source_kw": row.get("source_mw", 0.0) * 1000.0,
                "source_with_battery_kw": row.get("source_with_battery_mw", 0.0) * 1000.0,
                "electric_load_kw": row.get("electric_load_mw", 0.0) * 1000.0,
                "building_hvac_electric_kw": row.get("building_mw", 0.0) * 1000.0,
                "hvac_mode": row.get("hvac_mode", "off"),
                "p_hvac_request_kw": row.get("p_hvac_request_kw", 0.0),
                "p_heat_kw": row.get("p_heat_kw", 0.0),
                "p_cool_kw": row.get("p_cool_kw", 0.0),
                "p_hvac_electric_kw": row.get("p_hvac_electric_kw", 0.0),
                "q_hvac": row.get("q_hvac", 0.0),
                "available_h2_power_kw": row.get("available_h2_power_mw", 0.0) * 1000.0,
                "p_ez_kw": row.get("p_ez_mw", 0.0) * 1000.0,
                "grid_purchase_kwh": row.get("grid_purchase_mwh", 0.0) * 1000.0,
                "p_curtail_kw": row.get("p_curtail_mw", 0.0) * 1000.0,
                "curtail_cost_yuan": row.get("curtail_cost_yuan", 0.0),
                "grid_purchase_cost_yuan": row.get("grid_purchase_cost_yuan", 0.0),
                "h2_violation_event": row.get("h2_violation_event", 0.0),
                "h2_lower_violation_event": row.get("h2_lower_violation_event", 0.0),
                "h2_upper_violation_event": row.get("h2_upper_violation_event", 0.0),
                "h2_lower_violation_cost_yuan": row.get("h2_lower_violation_cost_yuan", 0.0),
                "h2_upper_violation_cost_yuan": row.get("h2_upper_violation_cost_yuan", 0.0),
                "h2_violation_cost_yuan": row.get("h2_violation_cost_yuan", 0.0),
                "battery_violation_event": row.get("battery_violation_event", 0.0),
                "battery_soc_lower_violation_event": row.get("battery_soc_lower_violation_event", 0.0),
                "battery_soc_upper_violation_event": row.get("battery_soc_upper_violation_event", 0.0),
                "battery_soc_violation_event": row.get("battery_soc_violation_event", 0.0),
                "battery_soc_lower_violation_cost_yuan": row.get("battery_soc_lower_violation_cost_yuan", 0.0),
                "battery_soc_upper_violation_cost_yuan": row.get("battery_soc_upper_violation_cost_yuan", 0.0),
                "battery_violation_cost_yuan": row.get("battery_violation_cost_yuan", 0.0),
                "comfort_violation_event": row.get("comfort_violation_event", 0.0),
                "comfort_violation_cost_yuan": row.get("comfort_violation_cost_yuan", 0.0),
                "carbon_cost_yuan": row.get("carbon_cost_yuan", 0.0),
                "carbon_credit_yuan": row.get("carbon_credit_yuan", 0.0),
                "scp_growth_revenue_yuan": row.get("scp_growth_revenue_yuan", 0.0),
                "scp_harvest_revenue_yuan": row.get("scp_harvest_revenue_yuan", 0.0),
                "bio_forced_replacement_event": row.get("bio_forced_replacement_event", 0.0),
                "bio_forced_replacement_cost_yuan": row.get("bio_forced_replacement_cost_yuan", 0.0),
                "bio_h2_violation_event": row.get("bio_h2_violation_event", 0.0),
                "bio_h2_lower_demand_violation_event": row.get("bio_h2_lower_demand_violation_event", 0.0),
                "bio_h2_upper_demand_violation_event": row.get("bio_h2_upper_demand_violation_event", 0.0),
                "bio_h2_lower_demand_violation_cost_yuan": row.get("bio_h2_lower_demand_violation_cost_yuan", 0.0),
                "bio_h2_upper_demand_violation_cost_yuan": row.get("bio_h2_upper_demand_violation_cost_yuan", 0.0),
                "bio_h2_violation_cost_yuan": row.get("bio_h2_violation_cost_yuan", 0.0),
                "total_revenue_yuan": row.get("total_revenue_yuan", 0.0),
                "total_cost_yuan": row.get("total_cost_yuan", 0.0),
                "step_profit_yuan": row.get("step_profit_yuan", 0.0),
                "co2_absorbed_mol": row.get("bio_co2_uptake_mol", 0.0),
                "co2_absorbed_t": row.get("bio_co2_absorption_t", 0.0),
                "grid_co2_emission_t": row.get("grid_co2_emission_t", 0.0),
                "net_co2_emission_t": row.get("net_co2_emission_t", 0.0),
                "T_room_C": row.get("T_room", 0.0),
                "T_wall_C": row.get("T_wall", 0.0),
                "scp_increment_g_protein": row.get("bio_dSCP_g_protein", 0.0),
                "scp_active_g_protein": scp_active,
                "scp_harvested_increment_g_protein": row.get("bio_harvested_SCP_g_protein", 0.0),
                "scp_cumulative_harvested_g_protein": scp_harvested_total,
                "scp_total_available_g_protein": scp_active + scp_harvested_total,
                "bio_X_gDW": row.get("bio_X_gDW_after_event", row.get("bio_X_gDW", 0.0)),
                "bio_H2_growth_fraction": row.get("bio_H2_growth_fraction", 0.0),
                "bio_H2_maintenance_fraction": row.get("bio_H2_maintenance_fraction", 0.0),
                "bio_replacement_event": row.get("bio_replacement_event", 0.0),
                "bio_replacement_reason": row.get("bio_replacement_reason", "none"),
                "bio_low_efficiency_replacement_event": row.get("bio_low_efficiency_replacement_event", 0.0),
                "bio_severe_h2_shortfall_replacement_event": row.get("bio_severe_h2_shortfall_replacement_event", 0.0),
                "bio_severe_h2_shortfall_hours": row.get("bio_severe_h2_shortfall_hours", 0.0),
                "battery_charge_kw": row.get("battery_charge_mw", 0.0) * 1000.0,
                "battery_discharge_kw": row.get("battery_discharge_mw", 0.0) * 1000.0,
                "battery_power_kw": row.get("battery_power_mw", 0.0) * 1000.0,
                "battery_soc_kwh": row.get("battery_soc_mwh", 0.0) * 1000.0,
                "battery_soc_fraction": row.get("battery_soc_fraction", 0.0),
                "h2_tank_mol": row.get("h2_tank_mol", 0.0),
                "h2_tank_violation_mol": row.get("h2_tank_violation_mol", 0.0),
                "h2_lower_violation_mol": row.get("h2_lower_violation_mol", 0.0),
                "h2_upper_violation_mol": row.get("h2_upper_violation_mol", 0.0),
                "ev_h2_out_mol_h": row.get("ev_h2_out_mol_h", 0.0),
                "h2_after_ev_mol": row.get("h2_after_ev_mol", 0.0),
                "bio_available_h2_mol_h": row.get("bio_available_h2_mol_h", 0.0),
                "bio_h2_feed_requested_mol_h": row.get("bio_h2_feed_requested_mol_h", 0.0),
                "bio_h2_feed_executed_mol_h": row.get("bio_h2_feed_executed_mol_h", 0.0),
                "bio_h2_actual_feed_mol_h": row.get("bio_h2_actual_feed_mol_h", 0.0),
                "bio_h2_load_mol_h": row.get("bio_h2_load_mol_h", 0.0),
                "bio_h2_uptake_mol": row.get("bio_h2_uptake_mol", 0.0),
                "bio_h2_survival_min_mol_h": row.get("bio_h2_survival_min_mol_h", 0.0),
                "bio_max_demand_h2_mol_h": row.get("bio_max_demand_h2_mol_h", 0.0),
                "bio_h2_survival_shortfall_mol_h": row.get("bio_h2_survival_shortfall_mol_h", 0.0),
                "bio_h2_biological_absorption_max_mol_h": row.get("bio_h2_biological_absorption_max_mol_h", 0.0),
                "bio_h2_lower_demand_violation_mol_h": row.get("bio_h2_lower_demand_violation_mol_h", 0.0),
                "bio_h2_upper_demand_violation_mol_h": row.get("bio_h2_upper_demand_violation_mol_h", 0.0),
                "bio_h2_physical_supply_max_mol_h": row.get("bio_h2_physical_supply_max_mol_h", 0.0),
                "reward": row.get("reward", 0.0),
                "step_cost": row.get("step_cost", 0.0),
            }
            writer.writerow(out)
    return path

EPISODE_REWARD_SUM_SPECS = [
    ("reward", "reward", 1.0),
    ("step_profit_yuan", "step_profit_yuan", 1.0),
    ("step_cost", "step_cost", 1.0),
    ("total_revenue_yuan", "total_revenue_yuan", 1.0),
    ("total_cost_yuan", "total_cost_yuan", 1.0),
    ("curtail_cost_yuan", "curtail_cost_yuan", 1.0),
    ("grid_purchase_cost_yuan", "grid_purchase_cost_yuan", 1.0),
    ("h2_lower_violation_cost_yuan", "h2_lower_violation_cost_yuan", 1.0),
    ("h2_upper_violation_cost_yuan", "h2_upper_violation_cost_yuan", 1.0),
    ("h2_violation_cost_yuan", "h2_violation_cost_yuan", 1.0),
    ("battery_soc_lower_violation_cost_yuan", "battery_soc_lower_violation_cost_yuan", 1.0),
    ("battery_soc_upper_violation_cost_yuan", "battery_soc_upper_violation_cost_yuan", 1.0),
    ("battery_violation_cost_yuan", "battery_violation_cost_yuan", 1.0),
    ("comfort_violation_cost_yuan", "comfort_violation_cost_yuan", 1.0),
    ("carbon_cost_yuan", "carbon_cost_yuan", 1.0),
    ("carbon_credit_yuan", "carbon_credit_yuan", 1.0),
    ("scp_growth_revenue_yuan", "scp_growth_revenue_yuan", 1.0),
    ("scp_harvest_revenue_yuan", "scp_harvest_revenue_yuan", 1.0),
    ("bio_forced_replacement_cost_yuan", "bio_forced_replacement_cost_yuan", 1.0),
    ("bio_h2_lower_demand_violation_cost_yuan", "bio_h2_lower_demand_violation_cost_yuan", 1.0),
    ("bio_h2_upper_demand_violation_cost_yuan", "bio_h2_upper_demand_violation_cost_yuan", 1.0),
    ("bio_h2_violation_cost_yuan", "bio_h2_violation_cost_yuan", 1.0),
    ("grid_purchase_kwh", "grid_purchase_mwh", 1000.0),
    ("curtail_kwh", "curtail_mwh", 1000.0),
    ("grid_co2_emission_t", "grid_co2_emission_t", 1.0),
    ("bio_co2_absorption_t", "bio_co2_absorption_t", 1.0),
    ("net_co2_emission_t", "net_co2_emission_t", 1.0),
    ("h2_tank_violation_mol", "h2_tank_violation_mol", 1.0),
    ("h2_lower_violation_mol", "h2_lower_violation_mol", 1.0),
    ("h2_upper_violation_mol", "h2_upper_violation_mol", 1.0),
    ("battery_charge_kw", "battery_charge_mw", 1000.0),
    ("battery_discharge_kw", "battery_discharge_mw", 1000.0),
]

def summarize_epoch_reward_components(episode, split_name, total_reward, records):
    if not records:
        raise ValueError("Cannot summarize an empty episode record list.")

    row = {
        "epoch": int(episode),
        "episode": int(episode),
        "split": split_name,
        "total_reward": float(total_reward),
        "mean_reward": float(total_reward) / max(len(records), 1),
        "hours": int(len(records)),
    }
    for export_key, source_key, scale in EPISODE_REWARD_SUM_SPECS:
        values = np.asarray([record[source_key] for record in records], dtype=np.float64) * scale
        row[f"{export_key}_sum"] = float(np.sum(values))
        row[f"{export_key}_mean"] = float(np.mean(values))

    last_record = records[-1]
    row["final_h2_tank_mol"] = float(last_record["h2_tank_mol"])
    row["final_battery_soc_kwh"] = float(last_record["battery_soc_mwh"]) * 1000.0
    row["final_battery_soc_fraction"] = float(last_record["battery_soc_fraction"])
    row["mean_bio_H2_growth_fraction"] = float(np.mean([r["bio_H2_growth_fraction"] for r in records]))
    row["mean_bio_starvation_level"] = float(np.mean([r["bio_starvation_level"] for r in records]))
    row["mean_p_ez_kw"] = float(np.mean([r["p_ez_mw"] for r in records])) * 1000.0
    row["mean_grid_purchase_kwh"] = float(np.mean([r["grid_purchase_mwh"] for r in records])) * 1000.0
    return row


def write_epoch_reward_history_csv(save_dir, split_name, rows):
    if not rows:
        return None
    path = os.path.join(save_dir, f"epoch_reward_components_{split_name}.csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_combined_epoch_reward_history_csv(save_dir, rows):
    if not rows:
        return None
    path = os.path.join(save_dir, "epoch_reward_components_all.csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_split_results(save_dir, split_name, scenario, records, reward, episode_rewards=None):
    result = records_to_arrays(records)
    if split_name in {"train", "test"}:
        save_reward_components_csv(save_dir, split_name, records)
    if split_name == "test":
        save_test_timeseries_details_csv(save_dir, records)
    payload = {
        "pv": scenario.pv,
        "wind": scenario.wind,
        "load": scenario.load,
        "t_out": scenario.t_out,
        "reward": np.asarray([reward], dtype=np.float32),
        **result,
    }
    if episode_rewards is not None:
        payload["episode_rewards"] = np.asarray(episode_rewards, dtype=np.float32)
    np.savez(os.path.join(save_dir, f"sac_microgrid_{split_name}_results.npz"), **payload)
    plot_bio_microgrid_results(
        scenario.pv,
        scenario.wind,
        result["battery_discharge_mw"],
        result["p_curtail_mw"],
        scenario.load,
        result["p_ez_mw"],
        result["building_mw"],
        result["battery_charge_mw"],
        result["h2_tank_mol"],
        result["T_room"],
        scenario.t_out,
        result["T_wall"],
        save_path=os.path.join(save_dir, f"SAC_{split_name}_overall_strategy.png"),
    )
    return result


def run_sac_optimization(configs, sac_cfg=None, save_dir=None):
    sac_cfg = sac_cfg or SACConfig()
    save_dir = save_dir or os.path.join(BASE_DIR, "SAC")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(sac_cfg.seed)

    full_data = load_scenario_data(configs)
    source_load_csv_path = os.path.join(save_dir, "source_load_match.csv")
    full_data, energy_summary = match_source_to_load_ratio(
        full_data, configs, bio_h2_feed_max_mol_h=5000.0, csv_path=source_load_csv_path
    )
    split_data, split_lengths = split_scenario_data(full_data)
    print(
        "Dataset split (hours): "
        f"total={len(full_data.ev_demand)}, train={split_lengths['train']}, "
        f"test={split_lengths['test']}, validation={split_lengths['validation']}"
    )
    print(
        "Source-load match: "
        f"source_before={energy_summary['source_mwh_before']:.3f} MWh, "
        f"source_after={energy_summary['source_mwh_after']:.3f} MWh, "
        f"load_equiv={energy_summary['total_load_equivalent_mwh']:.3f} MWh, "
        f"ratio={energy_summary['source_mwh_after'] / max(energy_summary['total_load_equivalent_mwh'], 1e-12):.3f}"
    )

    train_data = split_data["train"]
    test_data = split_data["test"]
    validation_data = split_data["validation"]
    norm_stats = build_norm_stats(train_data)

    env = MicrogridBioSACEnv(
        configs,
        train_data.pv,
        train_data.wind,
        train_data.load,
        train_data.ev_demand,
        train_data.t_out,
        train_data.irradiance,
        norm_stats=norm_stats,
    )
    validation_env = MicrogridBioSACEnv(
        configs,
        validation_data.pv,
        validation_data.wind,
        validation_data.load,
        validation_data.ev_demand,
        validation_data.t_out,
        validation_data.irradiance,
        norm_stats=norm_stats,
    )
    test_env = MicrogridBioSACEnv(
        configs,
        test_data.pv,
        test_data.wind,
        test_data.load,
        test_data.ev_demand,
        test_data.t_out,
        test_data.irradiance,
        norm_stats=norm_stats,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env.state_dim, env.action_dim, sac_cfg, device)
    replay_buffer = ReplayBuffer(sac_cfg.buffer_size, env.state_dim, env.action_dim, device)

    episode_rewards = []
    epoch_train_reward_rows = []
    epoch_test_reward_rows = []
    epoch_all_reward_rows = []
    noise_std = sac_cfg.exploration_noise
    best_reward = -np.inf
    best_records = None

    for episode in range(1, sac_cfg.episodes + 1):
        window_start = 0
        window_hours = len(train_data.pv)
        if sac_cfg.random_train_window:
            window_hours = int(sac_cfg.episode_window_hours)
            if window_hours <= 0:
                raise ValueError("episode_window_hours must be > 0")
            if window_hours > len(train_data.pv):
                raise ValueError(
                    "episode_window_hours cannot exceed the training set length."
                )
            max_start = len(train_data.pv) - window_hours
            window_start = int(np.random.randint(0, max_start + 1))
            train_episode_data = slice_scenario_window(
                train_data, window_start, window_hours
            )
            env.set_scenario_data(train_episode_data)
        else:
            env.set_scenario_data(train_data)

        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            replay_buffer.store(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            if replay_buffer.size >= sac_cfg.batch_size:
                agent.learn(replay_buffer)

        episode_rewards.append(total_reward)
        train_episode_records = list(env.records)
        train_epoch_row = summarize_epoch_reward_components(
            episode, "train", total_reward, train_episode_records
        )
        test_epoch_reward, test_epoch_records = evaluate_policy(test_env, agent)
        test_epoch_row = summarize_epoch_reward_components(
            episode, "test", test_epoch_reward, test_epoch_records
        )
        epoch_train_reward_rows.append(train_epoch_row)
        epoch_test_reward_rows.append(test_epoch_row)
        epoch_all_reward_rows.extend([train_epoch_row, test_epoch_row])
        write_epoch_reward_history_csv(save_dir, "train", epoch_train_reward_rows)
        write_epoch_reward_history_csv(save_dir, "test", epoch_test_reward_rows)
        write_combined_epoch_reward_history_csv(save_dir, epoch_all_reward_rows)

        noise_std = max(sac_cfg.min_noise, noise_std * sac_cfg.noise_decay)
        validation_reward, validation_records = evaluate_policy(validation_env, agent)
        if validation_reward > best_reward:
            best_reward = validation_reward
            best_records = validation_records
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_best.pth"))

        print(
            f"Episode {episode:04d} | window_start={window_start} "
            f"| window_hours={window_hours} | train_reward={total_reward:.3f} "
            f"| test_reward={test_epoch_reward:.3f} "
            f"| validation_reward={validation_reward:.3f} | noise={noise_std:.3f}"
        )

    if best_records is None:
        best_reward, best_records = evaluate_policy(validation_env, agent)

    best_actor_path = os.path.join(save_dir, "actor_best.pth")
    if os.path.exists(best_actor_path):
        agent.actor.load_state_dict(torch.load(best_actor_path, map_location=device))

    env.set_scenario_data(train_data)
    train_reward, train_records = evaluate_policy(env, agent)
    validation_reward, validation_records = evaluate_policy(validation_env, agent)
    test_reward, test_records = evaluate_policy(test_env, agent)

    train_result = save_split_results(
        save_dir, "train", train_data, train_records, train_reward, episode_rewards
    )
    validation_result = save_split_results(
        save_dir, "validation", validation_data, validation_records, validation_reward
    )
    test_result = save_split_results(save_dir, "test", test_data, test_records, test_reward)

    np.savez(
        os.path.join(save_dir, "sac_split_summary.npz"),
        train_hours=np.asarray([split_lengths["train"]], dtype=np.int32),
        test_hours=np.asarray([split_lengths["test"]], dtype=np.int32),
        validation_hours=np.asarray([split_lengths["validation"]], dtype=np.int32),
        train_reward=np.asarray([train_reward], dtype=np.float32),
        validation_reward=np.asarray([validation_reward], dtype=np.float32),
        test_reward=np.asarray([test_reward], dtype=np.float32),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
        epoch_train_rewards=np.asarray(
            [row["total_reward"] for row in epoch_train_reward_rows], dtype=np.float32
        ),
        epoch_test_rewards=np.asarray(
            [row["total_reward"] for row in epoch_test_reward_rows], dtype=np.float32
        ),
        **{key: np.asarray([value], dtype=np.float32) for key, value in energy_summary.items()},
    )
    return {"train": train_result, "validation": validation_result, "test": test_result}, episode_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the microgrid plus HOB-SCP bio-load scenario with SAC.")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--episode-window-hours", type=int, default=72)
    parser.add_argument(
        "--no-random-train-window",
        action="store_true",
        help="Use the full training split for every episode instead of random windows.",
    )
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "SAC"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SACConfig(
        episodes=args.episodes,
        seed=args.seed,
        random_train_window=not args.no_random_train_window,
        episode_window_hours=args.episode_window_hours,
    )
    run_sac_optimization(Configs(), cfg, args.save_dir)
