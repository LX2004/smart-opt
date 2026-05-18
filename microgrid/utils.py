import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def data_trans_scale(data, ref_data, ratio):
    data_sum = torch.sum(data)
    ref_data_sum = torch.sum(ref_data)
    data = data * ref_data_sum / data_sum * ratio
    return 

def get_PV_data():
    df_pv_actual = pd.read_excel('光伏实际出力.xlsx', skiprows = lambda x: x>0 and x<479972, nrows=24*30)
    pv_actual_list = df_pv_actual['值'].tolist()
    # print(df_pv_actual.head)
    return pv_actual_list

def get_wind_data():
    df_wind_actual = pd.read_excel('风电实际出力.xlsx', skiprows = lambda x: x>0 and x<448332, nrows=24*30)
    wind_actual_list = df_wind_actual['值'].tolist()
    # print(df_wind_actual.head)
    return wind_actual_list

def get_load_data():
    df = pd.read_csv('yearly_load_data.csv', skiprows = lambda x: x>0 and x<2, nrows=24*30)
    load_list = df['total_load_mw'].tolist()
    # print(df.head)
    return load_list

def get_ev_data():
    df = pd.read_csv('UrbanEV/data/volume-11kW.csv')
    list = df.iloc[2938:2938+30*24, 1:8].sum(axis=1).tolist()
    return list

def get_Temp_data():
    df = pd.read_csv('/home/liruyuan/my_proj/microgrid/t_amt_processed.csv')
    list = df.iloc[:,-1].tolist()
    return list

def get_weather_data(file_path):
    """
    使用 pandas 读取气象数据并返回四组列表
    """
    # 1. 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 2. 提取各列数据并转换为列表
    # 假设 CSV 列名为：temperature_2m (°C), wind_speed_10m (km/h), 
    # wind_speed_100m (km/h), global_tilted_irradiance (W/m²)
    
    # 注意：如果列名中包含空格或单位，建议使用索引或精确匹配列名
    temp_list = df.iloc[:, 1].tolist()      # 第2列：温度
    wind10_list = df.iloc[:, 2].tolist()    # 第3列：10m风速
    wind100_list = df.iloc[:, 3].tolist()   # 第4列：100m风速
    radiation_list = df.iloc[:, 4].tolist() # 第5列：辐照度
    
    return temp_list, wind10_list, wind100_list, radiation_list

def plot_results(pv, wind, fc, curtail, load, ez, building, h2, T_room, T_out, T_wall, save_path="/home/liruyuan/my_proj/microgrid/Figures/overall_strategy.png"):
    # 1. 自动提取并创建文件夹
    folder = os.path.dirname(save_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    # 确保数据是 numpy 数组，方便进行加法和负号操作
    pv = np.array(pv) 
    wind = np.array(wind) 
    fc = np.array(fc)
    load = np.array(load)
    ez = np.array(ez)
    curtail = np.array(curtail)
    building = np.array(building)

    hours = np.arange(len(pv))
    # 创建画布
    fig, ax = plt.subplots(3, 1, figsize=(30, 10))

    # --- 绘制正值部分 (供能堆叠) ---
    # 底层是 PV，中间是 Wind，顶层是 FC
    ax[0].bar(hours, pv, label='PV', color="#fa0505", edgecolor='black', linewidth=0.05)
    ax[0].bar(hours, wind, bottom=pv, label='Wind', color="#0095ff", edgecolor='black', linewidth=0.05)
    ax[0].bar(hours, fc, bottom=pv+wind, label='Fuel Cell', color="#ffae00", edgecolor='black', linewidth=0.05)

    # --- 绘制负值部分 (耗能与弃电堆叠) ---
    # 将 load, ez, curtail 转为负数
    neg_load = -load
    neg_ez = -ez
    neg_curtail = -curtail
    neg_building = -building

    # 底层是 Load，往下载是 Electrolyzer，再往下载是 Curtail
    ax[0].bar(hours, neg_load, label='Load', color='black', edgecolor='black', linewidth=0.05)
    ax[0].bar(hours, neg_ez, bottom=neg_load, label='Electrolyzer', color="#c300ff", edgecolor='black', linewidth=0.05)
    ax[0].bar(hours, neg_curtail, bottom=neg_load+neg_ez, label='Curtail', color="#00ff4c", edgecolor='black', linewidth=0.05)
    ax[0].bar(hours, neg_building, bottom=neg_load+neg_ez+neg_curtail, label='Building', color="#00D9FF9F", edgecolor='black', linewidth=0.05)

    # --- 装饰图表 ---
    ax[0].axhline(0, color='black', linewidth=1.5) # 画出 0 刻度基准线
    ax[0].set_title('Overall Energy Management Strategy', fontsize=16)
    ax[0].set_xlabel('Time (Hour)', fontsize=12)
    ax[0].set_ylabel('Power (MW)', fontsize=12)
    ax[0].grid(True, linestyle='--', alpha=0.3)
    
    # 将图例放在外面防止遮挡
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0.)

    ax[1].plot(h2, label='h2', color='blue', linewidth = 2)
    ax[1].set_title('h2 tank volume')
    ax[1].set_xlabel('Time (Hour)', fontsize=12)
    ax[1].set_ylabel('mol', fontsize=12)

    ax[2].plot(T_room, label='T_room', color='blue', linewidth=2.0)
    ax[2].plot(T_out, label='T_out', color='green', linewidth=2.0)
    ax[2].plot(T_wall, label='T_wall', color='black', linewidth=2.0)
    ax[2].set_xlabel('Time (Hour)', fontsize=12)
    ax[2].set_ylabel('celcius', fontsize=12)
    ax[2].legend()


    # 3. 保存并关闭
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"结果图片已成功保存至: {save_path}")