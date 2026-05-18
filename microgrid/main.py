from eq import PV, hydrogen_storage, fuel_cell, electrolytic_tank_ALK, electrolytic_tank_PEM, ElectrolyzerSystem, \
                smart_building, wind_turbine

from utils import get_PV_data, get_wind_data, get_load_data, plot_results, get_ev_data, get_Temp_data, get_weather_data
import numpy as np
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
import logging

logging.basicConfig(level=logging.INFO)

# export LD_LIBRARY_PATH=/var/conda/envs/hydrogen/lib:$LD_LIBRARY_PATH

class Configs:
    def __init__(self):
        # --- 1. 光伏系统参数 (PV) ---
        self.pv_num = 4e5              # 数量
        self.std_irradiance = 1000    # 标准辐照度
        self.temperature_coeff = -0.004 # 温度系数
        self.std_temp = 25            # 标准温度
        self.rated_power = 300        # 单块额定功率 (W)

        # --- 2. 储氢系统参数 (Hydrogen Storage) ---
        self.hydrogen_storage_pressure_max = 30.0 # MPa
        self.hydrogen_storage_pressure_min = 2.0
        self.ideal_gas_constant = 8.314
        self.hydrogen_storage_temp = 298.15 # K
        self.hydrogen_tank_vol = 5.0        # m^3
        self.hydrogen_tank_max_pressure = 35.0
        self.h2_tank_mol_min = 4e3
        self.h2_tank_mol_max = 6e4
        self.h2_tank_num = 400

        # --- 3. 电解槽与燃料电池参数 ---
        self.num_cell = 2e3          # 电池堆叠数
        self.area = 1000              # 反应面积 cm^2
        self.fuel_cell_num = 5e5       # 燃料电池数
        self.fuel_cell_active_area = 50

        self.ev2fcev_ratio_mol = 1/15 * 500
        self.human_comfort_temp = 24
        self.building_num = 100
 
        self.wind_turbine_num = 100

# --- 实例化并使用 ---
configs = Configs()

pv_data, wind_data, load_data, ev_demand = get_PV_data(), get_wind_data(), get_load_data(), get_ev_data()

t_data_C, w10_data_km_h, w100_data_km_h, rad_data_W_m2 = get_weather_data('open-meteo-47.98N104.95E1092m.csv')
# T_out =  get_Temp_data()
t_data_C, w10_data_km_h, w100_data_km_h, rad_data_W_m2 = t_data_C[:24*30], w10_data_km_h[:24*30], w100_data_km_h[:24*30], rad_data_W_m2[:24*30]

h2_trans = hydrogen_storage(configs)
fuel_cell_trans = fuel_cell(configs)
ez_PEM_trans = electrolytic_tank_PEM(configs)
building = smart_building()
pv_power = PV(configs)
wind_turbine_power = wind_turbine(configs)

pv_data = [pv_power(rad_data_W_m2[i], t_data_C[i]) * 1e-6 for i in range(len(rad_data_W_m2))]
wind_data = [wind_turbine_power(w100_data_km_h[i]) * 10 /36 for i in range(len(w100_data_km_h))]

# --- 第一步：预计算 (把物理模型转化为线性系数) ---
# 我们用你的类模拟一下：输入 1000W 功率，看产多少氢
ez_phys = ElectrolyzerSystem(configs)
sample_power = 1000
h2_flow_sample = ez_phys.step(sample_power) 
    
# 得到线性效率系数：mol/s 每瓦 (W)
# 这一步把复杂的物理公式变成了 MIP 能听懂的“比例”
ez_efficiency = h2_flow_sample / sample_power 

def run_microgrid_optimization_pyomo(configs, wind_data, pv_data, load_data):
    model = ConcreteModel()
    model.T = Set(initialize=range(len(wind_data)))

    # --- 变量定义 ---
    model.p_ez_mw = Var(model.T, bounds=(0, 50)) # 调大到 5MW，适配系统
    model.h2_tank_mol = Var(model.T, bounds=(configs.h2_tank_mol_min * configs.h2_tank_num, configs.h2_tank_mol_max * configs.h2_tank_num))
    model.p_curtail_mw = Var(model.T, bounds=(0, None)) # 弃电桶
    model.fuel_cell_h2_mol_s = Var(model.T, bounds=(0, None))
    model.fuel_cell_power_mw = Var(model.T, bounds=(0, None))
    model.T_wall = Var(model.T, bounds=(-20, 40), doc="墙体内部温度")
    model.P_hvac = Var(model.T, bounds=(0, 100), doc="空调电功率 kW")    
    model.fuel_cell_current_density_A_cm2 = Var(model.T, bounds=(0, 1.8))
    model.T_in = Var(model.T)


    # --- 目标函数 ---
    def objective_rule(model):
        curtail = sum(model.p_curtail_mw[t] for t in model.T)
        human_comfort = sum((model.T_in[t] - configs.human_comfort_temp)**2 for t in model.T)
        return curtail + human_comfort
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # --- 约束条件 ---

    # 1. 功率平衡
    def power_balance_rule(model, t):
        supply = wind_data[t] + pv_data[t] + model.fuel_cell_power_mw[t]
        demand = load_data[t] + model.p_ez_mw[t] + model.p_curtail_mw[t] + model.P_hvac[t] * 1e-3 * configs.building_num
        return supply == demand
    model.power_balance = Constraint(model.T, rule=power_balance_rule)

    # 2. 燃料电池特性 (拆解为多条约束)
    # 约束 A: 电流密度与氢气量的等式
    def fc_i_rule(model, t):
        return model.fuel_cell_current_density_A_cm2[t] == fuel_cell_trans.PEMFC_output_current_density_A_cm2(model.fuel_cell_h2_mol_s[t])
    model.fc_i_con = Constraint(model.T, rule=fc_i_rule)

    # 约束 B: 电流密度与电压的等式
    # def fc_v_rule(model, t):
    #     return model.fuel_cell_voltage_V[t] == fuel_cell_trans.PEMFC_output_voltage_V(model.fuel_cell_current_density_A_cm2[t])
    # model.fc_v_con = Constraint(model.T, rule=fc_v_rule)

    # 约束 C: 功率计算
    def fc_p_rule(model, t):
        # P = V * I * Area * Num
        fc_v = fuel_cell_trans.PEMFC_output_voltage_V(model.fuel_cell_current_density_A_cm2[t])
        return model.fuel_cell_power_mw[t] == fc_v * model.fuel_cell_current_density_A_cm2[t] * configs.fuel_cell_active_area * configs.fuel_cell_num * 1e-6
    model.fc_p_con = Constraint(model.T, rule=fc_p_rule)

    # 3. 储氢动态平衡
    def hydrogen_balance_rule(model, t):
        h2_in = model.p_ez_mw[t] * ez_efficiency * 1e6
        h2_out = model.fuel_cell_h2_mol_s[t] + ev_demand[t] * configs.ev2fcev_ratio_mol * 1e-3 
        if t == 0:
            return model.h2_tank_mol[t] == (configs.h2_tank_mol_min + configs.h2_tank_mol_max) * 0.5 * configs.h2_tank_num + (h2_in - h2_out) * 3600
        return model.h2_tank_mol[t] == model.h2_tank_mol[t-1] + (h2_in - h2_out) * 3600
    model.hydrogen_balance = Constraint(model.T, rule=hydrogen_balance_rule)

    def hydrogen_cycle_rule(model):
        # 找到最后一个时间步的索引
        last_t = max(model.T)
        first_t = min(model.T)
        # 始末平衡
        return model.h2_tank_mol[first_t] == model.h2_tank_mol[last_t]
    model.hydrogen_balance_cycle = Constraint(rule=hydrogen_cycle_rule)

    def air_temp_rule(model, t):
            if t == min(model.T):
                return model.T_in[t] == 24.0 # 初始室内温
            
            # 简化：Q_gain = 传热 + 太阳 + 人体 + 空调出力
            q_from_wall = (model.T_wall[t-1] - model.T_in[t-1]) / building.r1
            q_from_out = (t_data_C[t-1] - model.T_in[t-1]) / building.rwind
            # q_internal = self.configs.Cp_people * N_people_data[t-1]
            q_hvac = building.COP * model.P_hvac[t-1] 
            
            return model.T_in[t] == model.T_in[t-1] + (1.0 / building.czone) * \
                   (q_from_wall + q_from_out + q_hvac)
    model.air_temp_con = Constraint(model.T, rule=air_temp_rule)

    def wall_temp_rule(model, t):
            if t == min(model.T):
                return model.T_wall[t] == 15.0 # 初始墙温
            
            q_out_to_wall = (t_data_C[t-1] - model.T_wall[t-1]) / building.r1
            q_in_to_wall = (model.T_in[t-1] - model.T_wall[t-1]) / building.r2
            I_solar = pv_data[t-1] * 1000 / 0.8 / 100 
            q_solar = building.Gi_solar * I_solar
            
            return model.T_wall[t] == model.T_wall[t-1] + (1.0 / building.c) * \
                   (q_out_to_wall + q_in_to_wall + q_solar)
    model.wall_temp_con = Constraint(model.T, rule=wall_temp_rule)


    # --- 求解 ---
    solver = SolverFactory('ipopt')
    solver.solve(model, tee=True) # tee=True 可以实时看到求解器的收敛情况
    
    optimized_p_ez = [value(model.p_ez_mw[t]) for t in model.T]
    optimized_p_curtail_mw = [value(model.p_curtail_mw[t]) for t in model.T]
    optimized_p_fuel_cell_power_mw = [value(model.fuel_cell_power_mw[t]) for t in model.T]
    h2_tank = [value(model.h2_tank_mol[t]) for t in model.T]
    building_mw = [value(model.P_hvac[t])*1e-3 * configs.building_num for t in model.T]
    T_room = [value(model.T_in[t]) for t in model.T]
    T_wall = [value(model.T_wall[t])for t in model.T]

    return optimized_p_ez, optimized_p_curtail_mw, optimized_p_fuel_cell_power_mw, h2_tank, building_mw, T_room, T_wall

optimized_p_ez, optimized_p_curtail_mw, optimized_p_fuel_cell_power_mw, h2_tank, optimized_building_mw, T_room, T_wall = \
    run_microgrid_optimization_pyomo(configs, wind_data, pv_data, load_data)

# print(optimized_building_mw)
# print(h2_tank)

plot_results(pv_data, wind_data, optimized_p_fuel_cell_power_mw, optimized_p_curtail_mw, load_data, optimized_p_ez, optimized_building_mw, h2_tank, T_room, t_data_C, T_wall)
