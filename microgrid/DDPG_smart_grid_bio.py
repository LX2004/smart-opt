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

from eq import ElectrolyzerSystem, PV, smart_building, wind_turbine
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
        self.h2_tank_num = 400

        self.num_cell = 2e3
        self.area = 1000

        self.ev2fcev_ratio_mol = 1 / 15 * 500
        self.human_comfort_temp = 24
        self.building_num = 100
        self.wind_turbine_num = 100
        self.source_to_load_ratio = 1.2
        self.grid_emission_factor_tco2_per_mwh = 0.5703
        self.co2_molar_mass_g_per_mol = 44.0095


@dataclass
class DDPGConfig:
    episodes: int = 120
    batch_size: int = 256
    buffer_size: int = 200000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    hidden_dim: int = 256
    exploration_noise: float = 0.15
    noise_decay: float = 0.995
    min_noise: float = 0.02
    seed: int = 8


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
    within a batch, and the batch is harvested/replaced when H2 growth fraction
    falls to 0.5 or lower, which is equivalent to maintenance fraction reaching
    0.5 or higher.
    """

    def __init__(
        self,
        fixed_X0_gdw=5.0,
        h2_growth_fraction_cutoff=0.5,
        min_runtime_before_replacement_h=24.0,
    ):
        self.fixed_X0_gdw = float(fixed_X0_gdw)
        self.params = HOBParams()
        self.config = HOBLoadConfig(
            maintenance_fraction_cutoff=1.0 - h2_growth_fraction_cutoff,
            min_runtime_before_replacement_h=min_runtime_before_replacement_h,
            auto_replace=True,
            restart_immediately_after_replacement=True,
            reset_on_shutdown=True,
        )
        self.reset()

    def reset(self):
        self.load = HOBHydrogenLoad(
            X0_gdw=self.fixed_X0_gdw,
            params=self.params,
            load_config=self.config,
            start_active=True,
        )

    def state(self):
        return self.load.state()

    def minimum_survival_h2_mol_h(self):
        state = self.load.state()
        if state["active"] <= 0.0:
            return 0.0
        return float(state["X_gDW"] * self.params.m_h2 / 1000.0)

    def last_growth_fraction(self):
        return float(self.load.state()["last_H2_growth_fraction"])

    def age_h(self):
        return float(self.load.state()["age_h"])

    def step(self, h2_feed_mol_h, o2_feed_mol_h, co2_feed_mol_h, time_h):
        return self.load.step(
            F_H2_mol_h=float(h2_feed_mol_h),
            F_O2_mol_h=float(o2_feed_mol_h),
            F_CO2_mol_h=float(co2_feed_mol_h),
            dt_h=1.0,
            load_on=True,
            force_replace=False,
            time_h=float(time_h),
        )


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


def _hydrogen_mol_s_to_electrolyzer_mw(h2_mol_s, ez_efficiency):
    return np.asarray(h2_mol_s, dtype=np.float32) / max(ez_efficiency * 1e6, 1e-12)


def _hydrogen_mol_h_to_electrolyzer_mw(h2_mol_h, ez_efficiency):
    return _hydrogen_mol_s_to_electrolyzer_mw(float(h2_mol_h) / 3600.0, ez_efficiency)


def match_source_to_load_ratio(data, configs, bio_h2_feed_max_mol_h, csv_path=None):
    ez_efficiency = ElectrolyzerSystem(configs).step(1000) / 1000
    source_mwh_before = float(np.sum(data.pv + data.wind))
    electric_load_mwh = float(np.sum(data.load))

    ev_h2_mol_s = data.ev_demand * configs.ev2fcev_ratio_mol * 1e-3
    ev_h2_electric_mw = _hydrogen_mol_s_to_electrolyzer_mw(ev_h2_mol_s, ez_efficiency)
    bio_h2_electric_mw = float(_hydrogen_mol_h_to_electrolyzer_mw(bio_h2_feed_max_mol_h, ez_efficiency))

    ev_h2_electric_mwh = float(np.sum(ev_h2_electric_mw))
    bio_h2_electric_mwh = float(bio_h2_electric_mw * len(data.load))
    total_load_mwh = electric_load_mwh + ev_h2_electric_mwh + bio_h2_electric_mwh
    target_source_mwh = float(configs.source_to_load_ratio * total_load_mwh)

    if source_mwh_before <= 0.0:
        raise ValueError("Total renewable source energy must be positive for source-load matching.")

    source_scale = target_source_mwh / source_mwh_before
    matched = ScenarioData(
        pv=(data.pv * source_scale).astype(np.float32),
        wind=(data.wind * source_scale).astype(np.float32),
        load=data.load,
        ev_demand=data.ev_demand,
        t_out=data.t_out,
        wind100=data.wind100,
        irradiance=data.irradiance,
    )
    
    source_after_mw = matched.pv + matched.wind
    total_load_equivalent_mw = data.load + ev_h2_electric_mw + bio_h2_electric_mw
    summary = {
        "source_to_load_ratio": float(configs.source_to_load_ratio),
        "source_scale": float(source_scale),
        "source_mwh_before": source_mwh_before,
        "source_mwh_after": float(np.sum(source_after_mw)),
        "electric_load_mwh": electric_load_mwh,
        "ev_h2_electric_mwh": ev_h2_electric_mwh,
        "bio_h2_electric_mwh": bio_h2_electric_mwh,
        "total_load_equivalent_mwh": total_load_mwh,
        "target_source_mwh": target_source_mwh,
        "bio_h2_feed_max_mol_h": float(bio_h2_feed_max_mol_h),
    }
    if csv_path is not None:
        folder = os.path.dirname(csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "hour",
                "pv_mw",
                "wind_mw",
                "source_mw",
                "electric_load_mw",
                "ev_h2_equivalent_electric_mw",
                "bio_h2_equivalent_electric_mw",
                "total_load_equivalent_mw",
                "target_source_to_load_ratio",
            ])
            for i in range(len(data.load)):
                writer.writerow([
                    i,
                    float(matched.pv[i]),
                    float(matched.wind[i]),
                    float(source_after_mw[i]),
                    float(data.load[i]),
                    float(ev_h2_electric_mw[i]),
                    float(bio_h2_electric_mw),
                    float(total_load_equivalent_mw[i]),
                    float(configs.source_to_load_ratio),
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


class MicrogridBioDDPGEnv:
    def __init__(self, configs, pv_data, wind_data, load_data, ev_demand, t_data_C, rad_data_W_m2):
        self.configs = configs
        self.pv = np.asarray(pv_data, dtype=np.float32)
        self.wind = np.asarray(wind_data, dtype=np.float32)
        self.load = np.asarray(load_data, dtype=np.float32)
        self.ev_demand = np.asarray(ev_demand, dtype=np.float32)
        self.ev_h2_out_mol_h = self.ev_demand * configs.ev2fcev_ratio_mol * 1e-3 * 3600.0
        self.ev_h2_out_max_mol_h = max(float(np.max(self.ev_h2_out_mol_h)), 1.0)
        self.t_out = np.asarray(t_data_C, dtype=np.float32)
        self.rad = np.asarray(rad_data_W_m2, dtype=np.float32)
        self.horizon = len(self.pv)

        self.building = smart_building()
        self.bio_load = FixedStrainHydrogenLoad(
            fixed_X0_gdw=5.0,
            h2_growth_fraction_cutoff=0.5,
            min_runtime_before_replacement_h=24.0,
        )
        self.ez_efficiency = ElectrolyzerSystem(configs).step(1000) / 1000

        self.h2_min = configs.h2_tank_mol_min * configs.h2_tank_num
        self.h2_max = configs.h2_tank_mol_max * configs.h2_tank_num
        self.h2_initial = (self.h2_min + self.h2_max) * 0.5

        self.p_ez_max_mw = 50.0
        self.p_hvac_max_kw = 100.0
        self.bio_h2_feed_max_mol_h = 0.80
        self.bio_o2_ratio = 0.40
        self.bio_co2_ratio = 0.125
        self.biomass_reward_weight = 2.0
        self.harvest_reward_weight = 2.0
        self.bio_starvation_penalty_weight = 20.0
        self.carbon_penalty_weight = 1.0
        self.battery_capacity_mwh = 50.0
        self.battery_power_max_mw = 25.0
        self.battery_efficiency = 0.9
        self.battery_initial_mwh = 0.5 * self.battery_capacity_mwh

        self.state_dim = 18
        self.action_dim = 4
        self.reset()

    def reset(self):
        self.t = 0
        self.h2_tank_mol = self.h2_initial
        self.T_in = 24.0
        self.T_wall = 15.0
        self.battery_soc_mwh = self.battery_initial_mwh
        self.bio_load.reset()
        self.records = []
        return self._state()

    def _safe_scale(self, value, arr):
        high = float(np.max(arr))
        low = float(np.min(arr))
        return (float(value) - low) / (high - low + 1e-6)

    def _available_power_for_h2_mw(self, idx, building_mw=0.0):
        source_mw = float(self.pv[idx] + self.wind[idx])
        electric_load_mw = float(self.load[idx] + building_mw)
        return max(0.0, source_mw - electric_load_mw)

    def _apply_battery_command(self, command_mw, charge_power_limit_mw=None):
        command_mw = float(np.clip(command_mw, -self.battery_power_max_mw, self.battery_power_max_mw))
        soc_before = self.battery_soc_mwh
        charge_mw = 0.0
        discharge_mw = 0.0

        if command_mw >= 0.0:
            discharge_mw = min(
                command_mw,
                self.battery_power_max_mw,
                self.battery_soc_mwh * self.battery_efficiency,
            )
            self.battery_soc_mwh -= discharge_mw / self.battery_efficiency
        else:
            charge_mw = min(-command_mw, self.battery_power_max_mw)
            if charge_power_limit_mw is not None:
                charge_mw = min(charge_mw, max(float(charge_power_limit_mw), 0.0))
            charge_mw = min(
                charge_mw,
                (self.battery_capacity_mwh - self.battery_soc_mwh) / self.battery_efficiency,
            )
            self.battery_soc_mwh += charge_mw * self.battery_efficiency

        self.battery_soc_mwh = float(np.clip(self.battery_soc_mwh, 0.0, self.battery_capacity_mwh))
        return {
            "battery_command_mw": command_mw,
            "battery_charge_mw": float(charge_mw),
            "battery_discharge_mw": float(discharge_mw),
            "battery_power_mw": float(discharge_mw - charge_mw),
            "battery_soc_before_mwh": float(soc_before),
            "battery_soc_mwh": float(self.battery_soc_mwh),
            "battery_soc_fraction": float(self.battery_soc_mwh / max(self.battery_capacity_mwh, 1e-6)),
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
                self._safe_scale(self.pv[idx], self.pv),
                self._safe_scale(self.wind[idx], self.wind),
                self._safe_scale(self.load[idx], self.load),
                self._safe_scale(self.t_out[idx], self.t_out),
                ev_h2_out_fraction,
            ],
            dtype=np.float32,
        )

    def _map_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled = (action + 1.0) * 0.5
        p_ez_mw = float(scaled[0] * self.p_ez_max_mw)
        bio_h2_feed_mol_h = float(scaled[1] * self.bio_h2_feed_max_mol_h)
        p_hvac_kw = float(scaled[2] * self.p_hvac_max_kw)
        battery_command_mw = float(action[3] * self.battery_power_max_mw)
        return p_ez_mw, bio_h2_feed_mol_h, p_hvac_kw, battery_command_mw

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

    def _bio_step(self, h2_feed_mol_h):
        row = self.bio_load.step(
            h2_feed_mol_h=h2_feed_mol_h,
            o2_feed_mol_h=h2_feed_mol_h * self.bio_o2_ratio,
            co2_feed_mol_h=h2_feed_mol_h * self.bio_co2_ratio,
            time_h=self.t + 1,
        )
        return {
            "bio_h2_feed_mol_h": float(row["H2_input_mol_h"]),
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
            "bio_cycle_id": float(row["cycle_id"]),
            "bio_age_h": float(row["age_h"]),
        }

    def step(self, action):
        idx = self.t
        p_ez_command_mw, bio_h2_feed_mol_h, p_hvac_kw, battery_command_mw = self._map_action(action)
        building_mw = p_hvac_kw * 1e-3 * self.configs.building_num

        supply = float(self.pv[idx] + self.wind[idx])
        electric_load_mw = float(self.load[idx] + building_mw)
        battery_charge_limit_mw = max(0.0, supply - electric_load_mw)
        battery = self._apply_battery_command(battery_command_mw, battery_charge_limit_mw)
        supply_with_battery_mw = supply + battery["battery_discharge_mw"]
        available_h2_power_mw = max(
            0.0,
            supply_with_battery_mw - electric_load_mw - battery["battery_charge_mw"],
        )
        p_ez_mw = min(p_ez_command_mw, available_h2_power_mw, self.p_ez_max_mw)
        p_ez_clipped_mw = max(0.0, p_ez_command_mw - p_ez_mw)
        demand = electric_load_mw + battery["battery_charge_mw"] + p_ez_mw
        surplus = supply_with_battery_mw - demand
        curtail_mw = max(surplus, 0.0)
        grid_purchase_mwh = max(-surplus, 0.0)

        bio_min_survival_h2_mol_h = self.bio_load.minimum_survival_h2_mol_h()
        h2_in_mol_h = p_ez_mw * self.ez_efficiency * 1e6 * 3600.0
        ev_h2_out_mol_h = float(self.ev_h2_out_mol_h[idx])
        h2_after_ev_mol = self.h2_tank_mol - ev_h2_out_mol_h
        bio_available_h2_mol_h = max(0.0, h2_after_ev_mol + h2_in_mol_h - self.h2_min)
        bio_h2_feed_requested_mol_h = bio_h2_feed_mol_h
        bio_h2_feed_mol_h = min(bio_h2_feed_requested_mol_h, bio_available_h2_mol_h)
        bio = self._bio_step(bio_h2_feed_mol_h)
        next_h2 = h2_after_ev_mol + h2_in_mol_h - bio["bio_h2_uptake_mol"]
        h2_violation = max(self.h2_min - next_h2, 0.0) + max(next_h2 - self.h2_max, 0.0)
        self.h2_tank_mol = float(np.clip(next_h2, self.h2_min, self.h2_max))
        executed_action = self._physical_to_action(
            p_ez_mw,
            bio["bio_h2_feed_mol_h"],
            p_hvac_kw,
            battery["battery_power_mw"],
        )

        q_from_wall = (self.T_wall - self.T_in) / self.building.r1
        q_from_out = (float(self.t_out[idx]) - self.T_in) / self.building.rwind
        q_hvac = self.building.COP * p_hvac_kw
        next_T_in = self.T_in + (q_from_wall + q_from_out + q_hvac) / self.building.czone

        q_out_to_wall = (float(self.t_out[idx]) - self.T_wall) / self.building.r1
        q_in_to_wall = (self.T_in - self.T_wall) / self.building.r2
        irradiance_proxy = self.pv[idx] * 1000 / 0.8 / 100
        q_solar = self.building.Gi_solar * irradiance_proxy
        next_T_wall = self.T_wall + (q_out_to_wall + q_in_to_wall + q_solar) / self.building.c

        self.T_in = float(np.clip(next_T_in, -20.0, 40.0))
        self.T_wall = float(np.clip(next_T_wall, -20.0, 40.0))

        comfort_cost = (self.T_in - self.configs.human_comfort_temp) ** 2
        weighted_comfort_cost = 0.1 * comfort_cost
        h2_cost = 1e-8 * h2_violation**2
        grid_co2_emission_t = grid_purchase_mwh * self.configs.grid_emission_factor_tco2_per_mwh
        bio_co2_absorption_t = bio["bio_co2_uptake_mol"] * self.configs.co2_molar_mass_g_per_mol / 1e6
        net_co2_emission_t = grid_co2_emission_t - bio_co2_absorption_t
        carbon_cost = self.carbon_penalty_weight * net_co2_emission_t
        biomass_reward = self.biomass_reward_weight * max(bio["bio_dX_gDW"], 0.0)
        harvest_reward = self.harvest_reward_weight * max(bio["bio_harvested_SCP_g_protein"], 0.0)
        bio_starvation_shortage_mol_h = max(0.0, bio_min_survival_h2_mol_h - bio["bio_h2_load_mol_h"])
        bio_starvation_cost = self.bio_starvation_penalty_weight * bio_starvation_shortage_mol_h
        step_cost = (
            curtail_mw
            + weighted_comfort_cost
            + h2_cost
            + carbon_cost
            + bio_starvation_cost
            - biomass_reward
            - harvest_reward
        )
        reward = -step_cost

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
                "grid_purchase_mwh": grid_purchase_mwh,
                "grid_co2_emission_t": grid_co2_emission_t,
                "bio_co2_absorption_t": bio_co2_absorption_t,
                "net_co2_emission_t": net_co2_emission_t,
                "carbon_cost": carbon_cost,
                "h2_tank_mol": self.h2_tank_mol,
                "building_mw": building_mw,
                "bio_min_survival_h2_mol_h": bio_min_survival_h2_mol_h,
                "bio_starvation_shortage_mol_h": bio_starvation_shortage_mol_h,
                "bio_starvation_cost": bio_starvation_cost,
                "ev_h2_out_mol_h": ev_h2_out_mol_h,
                "h2_after_ev_mol": h2_after_ev_mol,
                "bio_available_h2_mol_h": bio_available_h2_mol_h,
                "bio_h2_feed_requested_mol_h": bio_h2_feed_requested_mol_h,
                "bio_h2_feed_executed_mol_h": bio["bio_h2_feed_mol_h"],
                "executed_action": executed_action,
                **bio,
                "T_room": self.T_in,
                "T_wall": self.T_wall,
                "curtail_cost": curtail_mw,
                "comfort_cost_raw": comfort_cost,
                "comfort_cost_weighted": weighted_comfort_cost,
                "h2_tank_violation_mol": h2_violation,
                "h2_tank_violation_cost": h2_cost,
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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(torch.cat([state, action], dim=-1))


class FeasibleActionProjector(nn.Module):
    def __init__(
        self,
        p_ez_max_mw,
        bio_h2_feed_max_mol_h,
        p_hvac_max_kw,
        building_num,
        battery_capacity_mwh,
        battery_power_max_mw,
        battery_efficiency,
        ez_efficiency,
        h2_min_mol,
        h2_max_mol,
        ev_h2_out_max_mol_h,
    ):
        super().__init__()
        self.p_ez_max_mw = float(p_ez_max_mw)
        self.bio_h2_feed_max_mol_h = float(bio_h2_feed_max_mol_h)
        self.p_hvac_max_kw = float(p_hvac_max_kw)
        self.building_num = float(building_num)
        self.battery_capacity_mwh = float(battery_capacity_mwh)
        self.battery_power_max_mw = float(battery_power_max_mw)
        self.battery_efficiency = float(battery_efficiency)
        self.ez_efficiency = float(ez_efficiency)
        self.h2_min_mol = float(h2_min_mol)
        self.h2_max_mol = float(h2_max_mol)
        self.ev_h2_out_max_mol_h = float(ev_h2_out_max_mol_h)

    @staticmethod
    def _min_tensor(value, limit):
        if not torch.is_tensor(limit):
            limit = torch.full_like(value, float(limit))
        return torch.minimum(value, limit)

    def forward(self, state, action):
        action = torch.clamp(action, -1.0, 1.0)
        scaled = (action + 1.0) * 0.5

        p_ez_command_mw = scaled[:, 0:1] * self.p_ez_max_mw
        bio_h2_command_mol_h = scaled[:, 1:2] * self.bio_h2_feed_max_mol_h
        p_hvac_kw = scaled[:, 2:3] * self.p_hvac_max_kw
        battery_command_mw = action[:, 3:4] * self.battery_power_max_mw

        available_no_building_mw = torch.clamp(state[:, 12:13], min=0.0) * self.p_ez_max_mw
        building_mw = p_hvac_kw * 1e-3 * self.building_num
        available_after_building_mw = torch.clamp(available_no_building_mw - building_mw, min=0.0)

        battery_soc_mwh = torch.clamp(state[:, 4:5], 0.0, 1.0) * self.battery_capacity_mwh
        discharge_request_mw = torch.clamp(battery_command_mw, min=0.0)
        charge_request_mw = torch.clamp(-battery_command_mw, min=0.0)
        battery_discharge_mw = self._min_tensor(
            self._min_tensor(discharge_request_mw, self.battery_power_max_mw),
            battery_soc_mwh * self.battery_efficiency,
        )
        battery_charge_mw = self._min_tensor(
            self._min_tensor(
                self._min_tensor(charge_request_mw, self.battery_power_max_mw),
                available_after_building_mw,
            ),
            (self.battery_capacity_mwh - battery_soc_mwh) / self.battery_efficiency,
        )
        battery_power_mw = battery_discharge_mw - battery_charge_mw

        available_h2_power_mw = torch.clamp(
            available_after_building_mw + battery_discharge_mw - battery_charge_mw,
            min=0.0,
        )
        p_ez_mw = self._min_tensor(
            self._min_tensor(p_ez_command_mw, available_h2_power_mw),
            self.p_ez_max_mw,
        )

        h2_tank_mol = torch.clamp(state[:, 3:4], 0.0, 1.0) * (
            self.h2_max_mol - self.h2_min_mol
        ) + self.h2_min_mol
        h2_in_mol_h = p_ez_mw * self.ez_efficiency * 1e6 * 3600.0
        ev_h2_out_mol_h = torch.clamp(state[:, 17:18], min=0.0) * self.ev_h2_out_max_mol_h
        bio_available_h2_mol_h = torch.clamp(
            h2_tank_mol - ev_h2_out_mol_h + h2_in_mol_h - self.h2_min_mol,
            min=0.0,
        )
        bio_h2_feed_mol_h = self._min_tensor(bio_h2_command_mol_h, bio_available_h2_mol_h)

        projected = torch.cat(
            [
                torch.clamp(2.0 * p_ez_mw / max(self.p_ez_max_mw, 1e-6) - 1.0, -1.0, 1.0),
                torch.clamp(
                    2.0 * bio_h2_feed_mol_h / max(self.bio_h2_feed_max_mol_h, 1e-6) - 1.0,
                    -1.0,
                    1.0,
                ),
                torch.clamp(2.0 * p_hvac_kw / max(self.p_hvac_max_kw, 1e-6) - 1.0, -1.0, 1.0),
                torch.clamp(battery_power_mw / max(self.battery_power_max_mw, 1e-6), -1.0, 1.0),
            ],
            dim=1,
        )
        return projected


class DDPGAgent:
    def __init__(self, state_dim, action_dim, cfg, device, action_projector=None):
        self.cfg = cfg
        self.device = device
        self.action_projector = action_projector
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_action(self, state, noise_std=0.0):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.action_projector is not None:
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.action_projector(state_tensor, action_tensor).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def project_action(self, states, actions):
        if self.action_projector is None:
            return torch.clamp(actions, -1.0, 1.0)
        return self.action_projector(states, actions)

    def learn(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            next_actions = self.project_action(next_states, self.actor_target(next_states))
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.project_action(states, self.actor(states))
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
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
    return {key: np.asarray([row[key] for row in records], dtype=np.float32) for key in records[0]}


def save_reward_components_csv(save_dir, split_name, records):
    reward_keys = [
        "hour",
        "reward",
        "step_cost",
        "curtail_cost",
        "comfort_cost_raw",
        "comfort_cost_weighted",
        "grid_purchase_mwh",
        "grid_co2_emission_t",
        "bio_co2_absorption_t",
        "net_co2_emission_t",
        "carbon_cost",
        "h2_tank_violation_mol",
        "h2_tank_violation_cost",
        "bio_starvation_cost",
        "battery_charge_mw",
        "battery_discharge_mw",
        "battery_soc_mwh",
        "battery_soc_fraction",
        "biomass_reward_component",
        "harvest_reward_component",
    ]
    path = os.path.join(save_dir, f"reward_components_{split_name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(reward_keys)
        for hour, row in enumerate(records):
            writer.writerow([hour if key == "hour" else float(row[key]) for key in reward_keys])
    return path


EPISODE_REWARD_SUM_KEYS = [
    "reward",
    "step_cost",
    "curtail_cost",
    "comfort_cost_weighted",
    "h2_tank_violation_cost",
    "carbon_cost",
    "bio_starvation_cost",
    "biomass_reward_component",
    "harvest_reward_component",
    "grid_purchase_mwh",
    "grid_co2_emission_t",
    "bio_co2_absorption_t",
    "net_co2_emission_t",
    "h2_tank_violation_mol",
    "battery_charge_mw",
    "battery_discharge_mw",
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
    for key in EPISODE_REWARD_SUM_KEYS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        row[f"{key}_sum"] = float(np.sum(values))
        row[f"{key}_mean"] = float(np.mean(values))

    last_record = records[-1]
    row["final_h2_tank_mol"] = float(last_record["h2_tank_mol"])
    row["final_battery_soc_mwh"] = float(last_record["battery_soc_mwh"])
    row["final_battery_soc_fraction"] = float(last_record["battery_soc_fraction"])
    row["mean_bio_H2_growth_fraction"] = float(np.mean([r["bio_H2_growth_fraction"] for r in records]))
    row["mean_bio_starvation_level"] = float(np.mean([r["bio_starvation_level"] for r in records]))
    row["mean_p_ez_mw"] = float(np.mean([r["p_ez_mw"] for r in records]))
    row["mean_grid_purchase_mwh"] = float(np.mean([r["grid_purchase_mwh"] for r in records]))
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
    np.savez(os.path.join(save_dir, f"ddpg_microgrid_{split_name}_results.npz"), **payload)
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
        save_path=os.path.join(save_dir, f"DDPG_{split_name}_overall_strategy.png"),
    )
    return result


def run_ddpg_optimization(configs, ddpg_cfg=None, save_dir=None):
    ddpg_cfg = ddpg_cfg or DDPGConfig()
    save_dir = save_dir or os.path.join(BASE_DIR, "DDPG_results")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(ddpg_cfg.seed)

    full_data = load_scenario_data(configs)
    source_load_csv_path = os.path.join(save_dir, "source_load_match.csv")
    full_data, energy_summary = match_source_to_load_ratio(
        full_data, configs, bio_h2_feed_max_mol_h=0.80, csv_path=source_load_csv_path
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

    env = MicrogridBioDDPGEnv(
        configs,
        train_data.pv,
        train_data.wind,
        train_data.load,
        train_data.ev_demand,
        train_data.t_out,
        train_data.irradiance,
    )
    validation_env = MicrogridBioDDPGEnv(
        configs,
        validation_data.pv,
        validation_data.wind,
        validation_data.load,
        validation_data.ev_demand,
        validation_data.t_out,
        validation_data.irradiance,
    )
    test_env = MicrogridBioDDPGEnv(
        configs,
        test_data.pv,
        test_data.wind,
        test_data.load,
        test_data.ev_demand,
        test_data.t_out,
        test_data.irradiance,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_projector = FeasibleActionProjector(
        p_ez_max_mw=env.p_ez_max_mw,
        bio_h2_feed_max_mol_h=env.bio_h2_feed_max_mol_h,
        p_hvac_max_kw=env.p_hvac_max_kw,
        building_num=configs.building_num,
        battery_capacity_mwh=env.battery_capacity_mwh,
        battery_power_max_mw=env.battery_power_max_mw,
        battery_efficiency=env.battery_efficiency,
        ez_efficiency=env.ez_efficiency,
        h2_min_mol=env.h2_min,
        h2_max_mol=env.h2_max,
        ev_h2_out_max_mol_h=env.ev_h2_out_max_mol_h,
    ).to(device)
    agent = DDPGAgent(env.state_dim, env.action_dim, ddpg_cfg, device, action_projector)
    replay_buffer = ReplayBuffer(ddpg_cfg.buffer_size, env.state_dim, env.action_dim, device)

    episode_rewards = []
    epoch_train_reward_rows = []
    epoch_test_reward_rows = []
    epoch_all_reward_rows = []
    noise_std = ddpg_cfg.exploration_noise
    best_reward = -np.inf
    best_records = None

    for episode in range(1, ddpg_cfg.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            replay_buffer.store(state, info["executed_action"], reward, next_state, float(done))
            state = next_state
            total_reward += reward
            if replay_buffer.size >= ddpg_cfg.batch_size:
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

        noise_std = max(ddpg_cfg.min_noise, noise_std * ddpg_cfg.noise_decay)
        validation_reward, validation_records = evaluate_policy(validation_env, agent)
        if validation_reward > best_reward:
            best_reward = validation_reward
            best_records = validation_records
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_best.pth"))

        if episode == 1 or episode % 10 == 0:
            print(
                f"Episode {episode:04d} | train_reward={total_reward:.3f} "
                f"| test_reward={test_epoch_reward:.3f} "
                f"| validation_reward={validation_reward:.3f} | noise={noise_std:.3f}"
            )

    if best_records is None:
        best_reward, best_records = evaluate_policy(validation_env, agent)

    best_actor_path = os.path.join(save_dir, "actor_best.pth")
    if os.path.exists(best_actor_path):
        agent.actor.load_state_dict(torch.load(best_actor_path, map_location=device))

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
        os.path.join(save_dir, "ddpg_split_summary.npz"),
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
    parser = argparse.ArgumentParser(description="Solve the microgrid plus HOB-SCP bio-load scenario with DDPG.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "DDPG_result"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DDPGConfig(episodes=args.episodes, seed=args.seed)
    run_ddpg_optimization(Configs(), cfg, args.save_dir)
