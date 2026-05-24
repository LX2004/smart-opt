import math
import numpy as np
from scipy.optimize import fsolve

class PV():
    def __init__(self, configs):
        self.pv_num = configs.pv_num
        self.std_irr = 1000
        self.temp_coeff = -0.0035
        self.std_temp = 25
        self.rated_power = 600

    def __call__(self, real_time_irr, temp):
        return self.pv_num * self.rated_power * real_time_irr / self.std_irr * \
            (1 + self.temp_coeff * (temp - self.std_temp))

class wind_turbine():

    def __init__(self, configs):
        self.v_cutin = 3.0
        self.v_cutout = 25
        self.v_rated = 11.4
        self.rated_p_mw = 5
        self.eq_num = configs.wind_turbine_num
    
    def __call__(self, V_wind_m_s):
        if V_wind_m_s < self.v_cutin or V_wind_m_s > self.v_cutout:
            return 0
        elif V_wind_m_s > self.v_rated:
            V_wind_m_s = self.v_rated
        return self.rated_p_mw * (V_wind_m_s**3 - self.v_cutin**3) / (self.v_rated**3 - self.v_cutin**3) * self.eq_num

class hydrogen_storage(): 

    def __init__(self, configs):
        self.storage_pressure_max = configs.hydrogen_storage_pressure_max
        self.storage_pressure_min = configs.hydrogen_storage_pressure_min

        self.ideal_gas_constant = configs.ideal_gas_constant
        self.storage_temp = configs.hydrogen_storage_temp

        self.tank_vol = configs.hydrogen_tank_vol
        self.max_pressure = configs.hydrogen_tank_max_pressure

        self.mol_min = configs.h2_tank_mol_min
        self.mol_max = configs.h2_tank_mol_max

    def new_hydrogen_state_mol(self, init_state, input_rate, output_rate, time_steps):
        net_hydrogen_rate = input_rate - output_rate
        return init_state + net_hydrogen_rate * time_steps
    
    def level_of_hydrogen(self, hydrogen_state):
        return self.ideal_gas_constant * self.storage_temp * hydrogen_state / self.tank_vol / self.max_pressure

class fuel_cell():
    def __init__(self, configs):
        self.fuel_cell_num = configs.fuel_cell_num
        self.open_circuit_voltage = 0.95
        self.fuel_eff = 0.95
        self.Faraday_const = 96485
        self.Tafel_slope = 1
        self.ohmic_slope = 0.2332
        self.concentration_polarization_slope = 0.5
        self.active_area = configs.fuel_cell_active_area
    
    def PEMFC_output_current_density_A_cm2(self, hydrogen_flow_rate_mol_s):
        return 2 * self.Faraday_const * hydrogen_flow_rate_mol_s * self.fuel_eff / self.fuel_cell_num / self.active_area
    
    # def PEMFC_output_voltage_V(self, current_density_A_cm2):
    #     # if current_density_A_cm2 <= 0.1:
    #     #     return self.open_circuit_voltage - self.Tafel_slope * current_density_A_cm2
    #     # elif current_density_A_cm2 > 0.1 and current_density_A_cm2 <= 1.6:
    #     #     return 0.85 - (current_density_A_cm2-0.1) * self.ohmic_slope
    #     # elif current_density_A_cm2 > 1.6 and current_density_A_cm2 <= 1.8:
    #     #     return 0.5 - (current_density_A_cm2-0.16) * self.concentration_polarization_slope
    #     # else:
    #     #     return('--------------Current Density out of boundary-----------------')
    #     return 0.85 - current_density_A_cm2 * self.ohmic_slope

    def PEMFC_output_voltage_V(self, current_density_A_cm2):
        """
        使用多项式拟合替代原有的分段逻辑。
        拟合公式: V = a*i^3 + b*i^2 + c*i + d
        """
        # 下面是基于你提供的斜率拟合出的近似系数 (建议用上面的脚本跑出精确值)
        # 这里的 a, b, c, d 是示例，运行脚本后替换为真实输出
        a = -0.09062228
        b = 0.24187205
        c = -0.42119861
        d = 0.91272547
        
        # 纯数学表达式，Pyomo 完美支持
        voltage = (a * current_density_A_cm2**3 + 
                   b * current_density_A_cm2**2 + 
                   c * current_density_A_cm2 + 
                   d)
        
        return voltage

class electrolytic_tank_ALK():
    def __init__(self, configs):
        self.reverse_voltage_V = 1.23
        self.ohmic_res_coeff = 0.4
        self.Tafel_slope_coeff = 0.10
        self.activation_offset = 0.5
        self.num_cell = configs.num_cell
        self.Faraday_const = 96485
        self.Faraday_eff = 0.95
        self.active_area = 1000

    def cell_voltage(self, current_density):
        single_cell_voltage = self.reverse_voltage_V + self.ohmic_res_coeff * current_density\
        + self.Tafel_slope_coeff * math.log10(self.activation_offset*current_density + 1)
        return self.num_cell * single_cell_voltage
    
    def hydrogen_production_rate_mol_s(self, current_density):
        current_A = current_density * self.active_area
        return self.Faraday_eff * self.num_cell * current_A / 2 / self.Faraday_const

    def cell_power(self, current_density):
        cell_voltage = self.cell_voltage(current_density)
        current = current_density * self.active_area
        return current * cell_voltage
    
class electrolytic_tank_PEM():
    def __init__(self, configs):
        self.reverse_voltage_V = 1.23
        self.ohmic_res_coeff = 0.2
        self.Tafel_slope_coeff = 0.06
        self.activation_offset = 2
        self.num_cell = configs.num_cell
        self.Faraday_const = 96485
        self.Faraday_eff = 0.95
        self.active_area = 1000

    def cell_voltage(self, current_density):
        single_cell_voltage = self.reverse_voltage_V + self.ohmic_res_coeff * current_density\
        + self.Tafel_slope_coeff * math.log10(self.activation_offset*current_density + 1)
        return self.num_cell * single_cell_voltage
    
    def hydrogen_production_rate_mol_s(self, current_density):
        current_A = current_density * self.active_area
        return self.Faraday_eff * self.num_cell * current_A / 2 / self.Faraday_const

    def cell_power(self, current_density):
        cell_voltage = self.cell_voltage(current_density)
        current = current_density * self.active_area
        return current * cell_voltage
    
class ElectrolyzerSystem():
    def __init__(self, configs):
        # 物理参数
        self.u_rev = 1.23
        self.r = 0.4
        self.s = 0.10
        self.t = 0.5
        
        # 结构参数
        self.num_cell = configs.num_cell
        self.area = configs.area  # cm^2
        self.faraday_eff = 0.95
        self.F = 96485

    def _power_equation(self, i, p_target):
        """定义方程：计算功率与目标功率的差值"""
        # 计算单体电压
        u_cell = self.u_rev + self.r * i + self.s * np.log10(self.t * i + 1)
        # 计算总功率 P = U_stack * I_total
        p_calc = (u_cell * self.num_cell) * (i * self.area)
        return p_calc - p_target

    def step(self, power_input_W):
        """
        核心方法：输入功率(W)，输出产氢流速(mol/s)
        """
        
        # 1. 反解电流密度 i
        # 使用 fsolve 寻找让 _power_equation 等于 0 的 i，初始猜测值设为 0.5
        i_sol = fsolve(self._power_equation, x0=0.5, args=(power_input_W,))
        i = max(0, i_sol[0]) # 确保电流密度不为负
        
        # 2. 根据求得的 i 计算产氢流速
        total_current = i * self.area
        h2_flow_mol_s = (self.faraday_eff * self.num_cell * total_current) / (2 * self.F)
        
        return h2_flow_mol_s
    
class smart_building():
    def __init__(self):
        # self.r1 = 0.116
        # self.r2 = 0.116
        # self.rwind = 6.55
        # self.czone = 1953.6
        # self.c = 314.7
        self.Gi_solar = 0.5
        self.COP = 3
        self.COP_heating = 3.0
        self.COP_cooling = 3.0
        self.comfort_temp_min = 22.0
        self.comfort_temp_max = 26.0
        self.pre_cooling_temp = 24.0
        self.r1 = 0.116
        self.r2 = 0.116
        self.rwind = 3
        self.czone = 500
        self.c = 100

    def split_hvac_power(self, T_room, requested_power_kw):
        requested_power_kw = max(float(requested_power_kw), 0.0)
        if T_room > self.comfort_temp_max:
            return {
                "mode": "cooling",
                "p_heat_kw": 0.0,
                "p_cool_kw": requested_power_kw,
                "p_hvac_electric_kw": requested_power_kw,
            }
        if T_room > self.pre_cooling_temp:
            return {
                "mode": "pre_cooling",
                "p_heat_kw": 0.0,
                "p_cool_kw": requested_power_kw,
                "p_hvac_electric_kw": requested_power_kw,
            }
        if T_room < self.comfort_temp_min:
            return {
                "mode": "heating",
                "p_heat_kw": requested_power_kw,
                "p_cool_kw": 0.0,
                "p_hvac_electric_kw": requested_power_kw,
            }
        return {
            "mode": "off",
            "p_heat_kw": 0.0,
            "p_cool_kw": 0.0,
            "p_hvac_electric_kw": 0.0,
        }

    def comfort_violation(self, T_room):
        if T_room < self.comfort_temp_min:
            return self.comfort_temp_min - T_room
        if T_room > self.comfort_temp_max:
            return T_room - self.comfort_temp_max
        return 0.0
    
class bio_factory():
    def __init__(self):
        self.h2_convert_kg_kgSCP = 0.4281
        self.power_convert_kWh_kgSCP = 8.8637
    def forward(self, h2, power):
        return 1/self.h2_convert_kg_kgSCP * h2 * 1 / 496.03,  1/self.power_convert_kWh_kgSCP * 1e-3