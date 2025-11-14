# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import pandas as pd
import networkx as nx

def angle_to_unit_vector(angle_degrees):# 调整为正北为0的vector转化
    # 将角度转换为弧度
    angle_radians = math.radians(90 - angle_degrees)

    # 计算单位向量的分量
    x = math.cos(angle_radians)
    y = math.sin(angle_radians)
    return np.array([x, y], dtype=float)

#计算边界期望方向性 r_exp
def expected_isotropic_directionality(delta_omega):
        """
        计算各向同性期望方向性（r_exp）
        delta_omega: 每个城市可达方向角域宽度 (弧度)
        """
        delta_omega = np.clip(delta_omega, 1e-6, 2 * np.pi)  # 防止除0
        return np.abs(np.sin(delta_omega / 2) / (delta_omega / 2))

def circular_span_min_arc(angles_rad: np.ndarray) -> float:
    """
    给定一组弧度角，返回覆盖所有角度的最小弧宽 ΔΩ（弧度）。
    角度为环形变量，使用“最大缺口法”：ΔΩ = 2π - max_gap
    """
    a = np.mod(angles_rad, 2*np.pi)
    if a.size == 0:
        return 2*np.pi
    a.sort()
    diffs = np.diff(np.r_[a, a[0] + 2*np.pi])  # 包含回绕差
    max_gap = diffs.max()
    return float(2*np.pi - max_gap)

#城市群内部每个城市流动的方向
def Local_vector_with_correction(Migration_file, output_file):
    """
    计算每个城市的流动方向性(UFD)，并加入边界效应校正
    """
    infor = pd.read_excel('../data/China_UA.xlsx')
    key_cities = list(infor['city'])
    value_cities = list(infor['id'])
    dictionary_cities = dict(zip(key_cities, value_cities))

    Migration_value = pd.read_csv(Migration_file, index_col=0)
    Migration_angle = pd.read_csv('../data/cities_angle.csv', index_col=0)
    Cleaned_UAdata = pd.read_excel('../data/Cleaned_UAdata.xlsx')
    UA_list = ['长三角', '哈长', '宁夏沿黄', '黔中', '珠三角', '长江中游',
               '山西中部', '天山北坡', '粤闽浙', '北部湾', '成渝', '中原',
               '辽中南', '关中平原', '呼包鄂榆', '滇中', '兰西', '山东半岛', '京津冀']

    infor.set_index('id', inplace=True)
    valid_ids = set(Cleaned_UAdata['id'])

    results = []

    for ua_name in UA_list:
        current_cities = [cid for cid in Migration_value.index if cid in valid_ids and infor.loc[cid, 'UA'] == ua_name]

        for cityA in current_cities:
            cityA_vector_sum = np.zeros(2)
            total_flow = 0.0

            for cityB in current_cities:
                flow = Migration_value.loc[int(cityA), str(cityB)]
                angle = Migration_angle.loc[int(cityA), str(cityB)]
                vector = angle_to_unit_vector(angle) * flow
                cityA_vector_sum += vector
                total_flow += flow

            # 合向量 (AMV)
            AMV_x, AMV_y = cityA_vector_sum/total_flow if total_flow != 0 else 0
            UFD_obs = np.linalg.norm(cityA_vector_sum)/100

            # 计算方向性强度（未校正）
            UFD_norm = np.linalg.norm(cityA_vector_sum) / total_flow if total_flow != 0 else 0

            # ----------------------------
            # 计算几何角域宽度 ΔΩ (简单近似)
            # ----------------------------
            # 假设每个城市的邻居角度分布范围即为 ΔΩ
            neighbor_angles = Migration_angle.loc[int(cityA), [str(c) for c in current_cities]].values
            neighbor_angles = np.radians(neighbor_angles[~np.isnan(neighbor_angles)])
            delta_omega = circular_span_min_arc(neighbor_angles)
            r_exp = expected_isotropic_directionality(delta_omega)
            UFD_corr = (UFD_norm - r_exp) / (1 - r_exp) if (1 - r_exp) != 0 else UFD_norm

            results.append({
                'City_ID': cityA,
                'City_Name': infor.loc[cityA, 'city'],
                'UA': ua_name,
                'AMV': np.array2string(cityA_vector_sum, precision=8, separator=' '),
                'AMV_x': AMV_x,
                'AMV_y': AMV_y,
                'UFD_obs': UFD_obs, #city_mrigrate_sum
                'UFD_norm': UFD_norm,
                'r_exp': r_exp,
                'UFD_corr': UFD_corr,
                'delta_omega(rad)': delta_omega
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    return df



#获取城市群内部每个城市流动的方向
if __name__ == '__main__':
    # print(global_vector)
    # df['global_vector'] = global_vector
    # df.to_csv('out/2021_spatial_indicator_global.csv')
    local_df=Local_vector_with_correction('../data/Mobility21_23.csv','./out_average/F_inUA_with_UFDcorr.csv')

#获取逐个季度城市群内部每个城市流动的方向
# if __name__ == '__main__':
#     input_dir = '../data/out_Q'
#     output_dir = './out_Q'
#     os.makedirs(output_dir, exist_ok=True)
#
#     for fname in os.listdir(input_dir):
#         if fname.endswith('.csv') and 'Migrate_out' in fname:
#             in_file = os.path.join(input_dir, fname)
#             out_file = os.path.join(output_dir, f'F_inUA_{fname}')
#             print(f"▶ 正在处理: {fname}")
#             try:
#                 Local_vector(Migration_file=in_file, output_file=out_file)
#                 print(f"✅ 输出完成: {out_file}")
#             except Exception as e:
#                 print(f"❌ 错误: {fname} — {e}")