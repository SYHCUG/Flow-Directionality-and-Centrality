import math
import random
import networkx as nx
import numpy as np
import pandas as pd

def mainpro(UA):
    # Step 0: 数据准备
    cities_df = pd.read_excel('../data/Cleaned_UAdata.xlsx')
    cities_df = cities_df[cities_df['UAID'] == UA]
    city_ids = cities_df["id"].astype(str).values
    Migration_flow = pd.read_csv('../data/Mobility21_23.csv', index_col=0)
    Migration_flow.index = Migration_flow.index.astype(str)
    Migration_flow.columns = Migration_flow.columns.astype(str)
    filtered_flowdata = Migration_flow.loc[city_ids, city_ids]
    # print(cities_df['UA'])
    UAnames=cities_df['UA'].tolist()
    UAname=UAnames[0]
    # print(UAname)
    flows = []
    # 遍历城市对，构建流动数据 只保留flowdata大于5的城市对
    for city_i in city_ids:
        for city_j in city_ids:
            flowdata = filtered_flowdata.loc[city_i, city_j]  # 使用 .loc
            if flowdata > 5:  # 检查流动值是否大于 5
                flows.append((city_i, city_j, flowdata))

    cities = dict(zip(cities_df['id'], zip(cities_df['Lat_WGS84'], cities_df['Lng_WGS84'])))
    city_data = dict(zip(cities_df['id'], zip(cities_df['Population'], cities_df['人均GDP'])))

    # Step 1: 使用 PageRank 算法识别潜在中心城市
    graph = nx.Graph()
    graph.add_weighted_edges_from(flows)
    pagerank = {node: value for node, value in nx.pagerank(graph, alpha=0.85).items() if node in city_ids}
    #
    # top_percentage = 0.2
    # top_n = int(len(pagerank) * top_percentage)
    # key_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]
#TODO 修改topnode逻辑 保证至少有两个

    top_percentage = 0.2
    top_n = max(2, int(len(pagerank) * top_percentage))  # 保证至少两个中心城市
    key_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]
    print("关键节点:", key_nodes)

    # Step 2: 计算中心城市权重 w_k
    def compute_center_weights(center_cities, city_data, alpha, beta):
        raw_weights = {}
        # print(city_data)
        for city in center_cities:

            # print(city_data.get(int(city)))
            population, per_capita_gdp = city_data.get(int(city))
            # print(population, per_capita_gdp )
            raw_weights[city] = alpha * population + beta * per_capita_gdp   # 人均GDP 转为总GDP
        total_weight = sum(raw_weights.values())
        return {city: weight / total_weight for city, weight in raw_weights.items()}

    #TODO 这里调整了超参数
    # 权重参数
    alpha = 0.5
    beta = 0.5
    center_weights = compute_center_weights(key_nodes, city_data, alpha, beta)
    print("中心城市权重 w_k:", center_weights)

    def angle_to_unit_vector(angle_degrees):#
        # 将角度转换为弧度
        angle_radians = math.radians(angle_degrees)
        # 计算单位向量的分量
        x = math.cos(angle_radians)
        y = math.sin(angle_radians)
        unit_vector = [x, y]
        return unit_vector

    def str_to_np(vector_str):
        # 去掉方括号并用空格分隔字符串以获取数值
        vector_list = vector_str.strip('[]').split()
        # print(vector_list)
        # 将字符串列表转换为浮点数列表
        vector_float = [float(i) for i in vector_list]
        return np.array(vector_float)

    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        cos_theta = dot_product / (v1_norm * v2_norm)
        theta = math.acos(cos_theta)

        return np.degrees(theta)


    def compute_angle(city_i, center_cities, center_weights):
        F_i = pd.read_csv("./out_average/F_inUA_origin.csv", index_col=0)
        F_i.set_index(keys='City_ID', inplace=True)
        cities_angle = pd.read_csv('../data/cities_angle.csv', index_col=0)

        # 计算加权夹角
        weighted_theta_sum = 0
        for center in center_cities:
            if int(city_i) == int(center):
                continue  # 跳过自己
            cityi2k_angle = cities_angle.loc[int(city_i), str(center)]
            cityi2k_vector = angle_to_unit_vector(cityi2k_angle)
            cityi_flow_vector = str_to_np(F_i.loc[int(city_i), 'vector_unit'])
            angle = angle_between_vectors(cityi2k_vector, cityi_flow_vector)
#TODo : 这里的角度计算可能需要调整 跟公式不符

            # gama = 1 - (angle / 360)
            # print(gama)
            # weighted_theta_sum += center_weights[center] * gama
            # print(weighted_theta_sum)
            cos_sim = np.dot(cityi2k_vector, cityi_flow_vector) / (
                        np.linalg.norm(cityi2k_vector) * np.linalg.norm(cityi_flow_vector))
            weighted_theta_sum += center_weights[center] * cos_sim

        return weighted_theta_sum


    # Step 3: 计算 γ_i
    city_gamma = {}

    # 遍历所有城市
    for city in cities:
        # 调用 compute_angle 函数，计算每个城市的 γ 值
        gamma_value = compute_angle(city, key_nodes, center_weights)
        # 将结果存储到字典中
        city_gamma[city] = gamma_value

    # city_gamma = {city: compute_angle(city, key_nodes, center_weights) for city in cities}
    city_gamma_df = pd.DataFrame(city_gamma.items(), columns=['City_ID', 'Gamma_Value'])
    gamma_mean = city_gamma_df['Gamma_Value'].mean()
    gamma_values = city_gamma_df['Gamma_Value']
    summary = {
        'UA': UAname,
        'mean': gamma_values.mean(),
        'mean_abs': gamma_values.abs().mean(),
        'std': gamma_values.std(),
        'positive_ratio': (gamma_values > 0).mean(),
        'negative_ratio': (gamma_values < 0).mean(),
    }
    output_path = f'./out_average/城市群中心性跳过自己/{UAname}_mean_{gamma_mean:.3f}.csv'
    # summary_df = pd.DataFrame([summary])

    # summary_df.to_csv(f'./out_test/gamma_summary_{UAname}.csv', index=False)
    city_gamma_df.to_csv(output_path, index=False)
    print("所有城市的 γ 值:", city_gamma_df['Gamma_Value'])

    return summary

if __name__ == '__main__':
    all_summaries = []
    for i in range(19):
        summary = mainpro(UA=i+1)  # 修改 mainpro 函数，使其返回 summary dict
        if summary:
            all_summaries.append(summary)
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('./out_average/gamma_summary_all_UAs.csv', index=False)


