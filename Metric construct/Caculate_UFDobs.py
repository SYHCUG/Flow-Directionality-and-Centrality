# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import pandas as pd
import networkx as nx

def angle_to_unit_vector(angle_degrees):
    angle_radians = math.radians(90 - angle_degrees)

    x = math.cos(angle_radians)
    y = math.sin(angle_radians)
    unit_vector = (x, y)
    return unit_vector


def Local_vector(Migration_file, output_file):
    infor = pd.read_excel('../data/China_UA.xlsx')

    key_cities = list(infor['city'])
    value_cities = list(infor['id'])
    dictionary_cities = dict(zip(key_cities, value_cities))
    df = pd.DataFrame(index=value_cities, columns=['local_vector'])
    Migration_value=pd.read_csv(Migration_file, index_col=0)
    # print(Migration_value.head())
    Migration_angle=pd.read_csv('../data/cities_angle.csv', index_col=0)
    # print(Migration_vector.head())
    Cleaned_UAdata = pd.read_excel('../data/Cleaned_UAdata.xlsx')
    UA_list = ['长三角', '哈长', '宁夏沿黄', '黔中', '珠三角', '长江中游',
               '山西中部', '天山北坡', '粤闽浙', '北部湾', '成渝', '中原',
               '辽中南', '关中平原', '呼包鄂榆', '滇中', '兰西', '山东半岛', '京津冀']
    infor.set_index('id',inplace=True)
    valid_ids = set(Cleaned_UAdata['id'])
    cityA_df = []
    cityA_id = []
    cityA_sum_df = []
    UA_df=[]
    cityA_vector=[]
    current_lng = []
    current_lat = []
    current_cityname=[]
    cityA_vector_unit= 0
    for i in range(len(UA_list)):
        current_cities=[]
        current_UA=UA_list[i]
        for cityA in Migration_value.index:
            if cityA not in valid_ids:
                continue
            cityA_UA=infor.loc[cityA,'UA']
            if cityA_UA==UA_list[i]:
                current_cities.append(cityA)
                current_cityname.append(infor.loc[cityA, 'city'])
                current_lng.append(infor.loc[cityA, 'Lng_WGS84'])
                current_lat.append( infor.loc[cityA, 'Lat_WGS84'])
        # print(current_cities)
        for cityA in current_cities:
            cityA_vector_sum = 0
            cityA_Migration_value_sum=Migration_value.loc[int(cityA)].sum()

            for cityB in current_cities:
 
                cityA2B_value = Migration_value.loc[int(cityA), str(cityB)]
                cityA2B_angle = Migration_angle.loc[int(cityA), str(cityB)]
                cityA2B_vector = list(angle_to_unit_vector(cityA2B_angle))

                cityA2B = cityA2B_value * np.array(cityA2B_vector)
                cityA_vector_sum += cityA2B 
                cityA_Migration_value_sum += cityA2B_value   

            magnitude = np.linalg.norm(cityA_vector_sum)

            if magnitude != 0:
                cityA_vector_unit = cityA_vector_sum / magnitude

            cityA_df.append(cityA_vector_unit)
            cityA_id.append(cityA)
            cityA_sum_df.append(magnitude)
            UA_df.append(current_UA)
            cityA_vector.append(cityA_vector_sum)


    data = {
        'City_ID': cityA_id,
        'city':current_cityname,
        'lng':current_lng,
        'lat':current_lat,
        'UA': UA_df,
        'vector_unit': cityA_df,
        'vector':cityA_vector,
        'city_mrigrate_sum': cityA_sum_df,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file)
    return df

if __name__ == '__main__':
    # print(global_vector)
    # df['global_vector'] = global_vector
    # df.to_csv('out/2021_spatial_indicator_global.csv')
    local_df=Local_vector('../data/Mobility21_23.csv','./out_average/F_inUA_origin.csv')
