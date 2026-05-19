import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eq import ElectrolyzerSystem, PV, fuel_cell, smart_building, wind_turbine
from utils import get_weather_data, plot_results


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
        self.fuel_cell_num = 5e5
        self.fuel_cell_active_area = 50

        self.ev2fcev_ratio_mol = 1 / 15 * 500
        self.human_comfort_temp = 24
        self.building_num = 100
        self.wind_turbine_num = 100


@dataclass
class DDPGConfig:
    episodes: int = 200
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

class MicrogridDDPGEnv:
    def __init__(self, configs, pv_data, wind_data, load_data, ev_demand, t_data_C, rad_data_W_m2):
        self.configs = configs
        self.pv = np.asarray(pv_data, dtype=np.float32)
        self.wind = np.asarray(wind_data, dtype=np.float32)
        self.load = np.asarray(load_data, dtype=np.float32)
        self.ev_demand = np.asarray(ev_demand, dtype=np.float32)
        self.t_out = np.asarray(t_data_C, dtype=np.float32)
        self.rad = np.asarray(rad_data_W_m2, dtype=np.float32)
        self.horizon = len(self.pv)

        self.building = smart_building()
        self.fuel_cell = fuel_cell(configs)
        self.ez_efficiency = ElectrolyzerSystem(configs).step(1000) / 1000

        self.h2_min = configs.h2_tank_mol_min * configs.h2_tank_num
        self.h2_max = configs.h2_tank_mol_max * configs.h2_tank_num
        self.h2_initial = (self.h2_min + self.h2_max) * 0.5

        self.p_ez_max_mw = 50.0
        self.p_hvac_max_kw = 100.0
        self.fc_i_max = 1.8
        self.fc_h2_max_mol_s = (
            self.fc_i_max
            * configs.fuel_cell_num
            * configs.fuel_cell_active_area
            / (2 * self.fuel_cell.Faraday_const * self.fuel_cell.fuel_eff)
        )

        self.state_dim = 10
        self.action_dim = 3
        self.reset()

    def reset(self):
        self.t = 0
        self.h2_tank_mol = self.h2_initial
        self.T_in = 24.0
        self.T_wall = 15.0
        self.records = []
        return self._state()

    def _safe_scale(self, value, arr):
        high = float(np.max(arr))
        low = float(np.min(arr))
        return (float(value) - low) / (high - low + 1e-6)

    def _state(self):
        idx = min(self.t, self.horizon - 1)
        h2_norm = (self.h2_tank_mol - self.h2_min) / (self.h2_max - self.h2_min + 1e-6)
        return np.asarray(
            [
                idx / max(self.horizon - 1, 1),
                np.sin(2 * np.pi * (idx % 24) / 24),
                np.cos(2 * np.pi * (idx % 24) / 24),
                h2_norm,
                (self.T_in + 20.0) / 60.0,
                (self.T_wall + 20.0) / 60.0,
                self._safe_scale(self.pv[idx], self.pv),
                self._safe_scale(self.wind[idx], self.wind),
                self._safe_scale(self.load[idx], self.load),
                self._safe_scale(self.t_out[idx], self.t_out),
            ],
            dtype=np.float32,
        )

    def _map_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled = (action + 1.0) * 0.5
        p_ez_mw = float(scaled[0] * self.p_ez_max_mw)
        fc_h2_mol_s = float(scaled[1] * self.fc_h2_max_mol_s)
        p_hvac_kw = float(scaled[2] * self.p_hvac_max_kw)
        return p_ez_mw, fc_h2_mol_s, p_hvac_kw

    def _fuel_cell_power_mw(self, h2_mol_s):
        current_density = self.fuel_cell.PEMFC_output_current_density_A_cm2(h2_mol_s)
        voltage = self.fuel_cell.PEMFC_output_voltage_V(current_density)
        power = (
            voltage
            * current_density
            * self.configs.fuel_cell_active_area
            * self.configs.fuel_cell_num
            * 1e-6
        )
        return max(float(power), 0.0), float(current_density)

    def step(self, action):
        idx = self.t
        p_ez_mw, fc_h2_mol_s, p_hvac_kw = self._map_action(action)
        fc_power_mw, fc_current_density = self._fuel_cell_power_mw(fc_h2_mol_s)
        building_mw = p_hvac_kw * 1e-3 * self.configs.building_num

        supply = float(self.pv[idx] + self.wind[idx] + fc_power_mw)
        demand = float(self.load[idx] + p_ez_mw + building_mw)
        surplus = supply - demand
        curtail_mw = max(surplus, 0.0)
        shortage_mw = max(-surplus, 0.0)

        h2_in = p_ez_mw * self.ez_efficiency * 1e6
        h2_out = fc_h2_mol_s + float(self.ev_demand[idx]) * self.configs.ev2fcev_ratio_mol * 1e-3
        next_h2 = self.h2_tank_mol + (h2_in - h2_out) * 3600
        h2_violation = max(self.h2_min - next_h2, 0.0) + max(next_h2 - self.h2_max, 0.0)
        self.h2_tank_mol = float(np.clip(next_h2, self.h2_min, self.h2_max))

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
        shortage_cost = 100.0 * shortage_mw**2
        h2_cost = 1e-8 * h2_violation**2
        fc_cost = 50.0 * max(fc_current_density - self.fc_i_max, 0.0) ** 2
        step_cost = curtail_mw + 0.1 * comfort_cost + shortage_cost + h2_cost + fc_cost

        self.records.append(
            {
                "p_ez_mw": p_ez_mw,
                "p_curtail_mw": curtail_mw,
                "shortage_mw": shortage_mw,
                "fuel_cell_power_mw": fc_power_mw,
                "fuel_cell_h2_mol_s": fc_h2_mol_s,
                "h2_tank_mol": self.h2_tank_mol,
                "building_mw": building_mw,
                "T_room": self.T_in,
                "T_wall": self.T_wall,
                "step_cost": step_cost,
            }
        )

        self.t += 1
        done = self.t >= self.horizon
        if done:
            cycle_error = abs(self.h2_tank_mol - self.h2_initial) / (self.h2_max - self.h2_min)
            reward = -(step_cost + 100.0 * cycle_error)
            next_state = self._state()
        else:
            reward = -step_cost
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
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.cfg = cfg
        self.device = device
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
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def learn(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
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


def save_split_results(save_dir, split_name, scenario, records, reward, episode_rewards=None):
    result = records_to_arrays(records)
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
    plot_results(
        scenario.pv,
        scenario.wind,
        result["fuel_cell_power_mw"],
        result["p_curtail_mw"],
        scenario.load,
        result["p_ez_mw"],
        result["building_mw"],
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
    split_data, split_lengths = split_scenario_data(full_data)
    print(
        "Dataset split (hours): "
        f"total={len(full_data.ev_demand)}, train={split_lengths['train']}, "
        f"test={split_lengths['test']}, validation={split_lengths['validation']}"
    )

    train_data = split_data["train"]
    test_data = split_data["test"]
    validation_data = split_data["validation"]

    env = MicrogridDDPGEnv(
        configs,
        train_data.pv,
        train_data.wind,
        train_data.load,
        train_data.ev_demand,
        train_data.t_out,
        train_data.irradiance,
    )
    validation_env = MicrogridDDPGEnv(
        configs,
        validation_data.pv,
        validation_data.wind,
        validation_data.load,
        validation_data.ev_demand,
        validation_data.t_out,
        validation_data.irradiance,
    )
    test_env = MicrogridDDPGEnv(
        configs,
        test_data.pv,
        test_data.wind,
        test_data.load,
        test_data.ev_demand,
        test_data.t_out,
        test_data.irradiance,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(env.state_dim, env.action_dim, ddpg_cfg, device)
    replay_buffer = ReplayBuffer(ddpg_cfg.buffer_size, env.state_dim, env.action_dim, device)

    episode_rewards = []
    noise_std = ddpg_cfg.exploration_noise
    best_reward = -np.inf
    best_records = None

    for episode in range(1, ddpg_cfg.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.store(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            if replay_buffer.size >= ddpg_cfg.batch_size:
                agent.learn(replay_buffer)

        episode_rewards.append(total_reward)
        noise_std = max(ddpg_cfg.min_noise, noise_std * ddpg_cfg.noise_decay)
        validation_reward, validation_records = evaluate_policy(validation_env, agent)
        if validation_reward > best_reward:
            best_reward = validation_reward
            best_records = validation_records
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_best.pth"))

        if episode == 1 or episode % 10 == 0:
            print(
                f"Episode {episode:04d} | train_reward={total_reward:.3f} "
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
    )
    return {"train": train_result, "validation": validation_result, "test": test_result}, episode_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the microgrid scenario with DDPG.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "DDPG_results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DDPGConfig(episodes=args.episodes, seed=args.seed)
    run_ddpg_optimization(Configs(), cfg, args.save_dir)
