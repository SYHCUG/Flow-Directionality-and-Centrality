# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import pandas as pd
import networkx as nx
from Cython.Compiler.PyrexTypes import c_int_ptr_type

infor=pd.read_excel('../data/China_UA.xlsx')

key_cities=list(infor['city'])
value_cities=list(infor['id'])
# dictionary_cities = dict(zip(value_cities,key_cities))
dictionary_cities = dict(zip(key_cities, value_cities))
df=pd.DataFrame(index=value_cities,columns=['global_vector'])
infor=infor.set_index('id')

def angle_to_unit_vector(angle_degrees):

    angle_radians = math.radians(90 - angle_degrees)

    x = math.cos(angle_radians)
    y = math.sin(angle_radians)
    unit_vector = (x, y)
    return unit_vector


def Global_vector(Migration_file):

    Migration_value = pd.read_csv(Migration_file, index_col=0)
    Migration_angle = pd.read_csv('../data/cities_angle.csv', index_col=0)

    unit_vectors = []       
    total_vectors = []     
    vector_magnitudes = []   
    city_name = []  

    for cityA in Migration_value.index:
        total_migration = Migration_value.loc[cityA].sum()
        total_vector = np.array([0.0, 0.0])

        for cityB in Migration_value.columns:
            migration_value = Migration_value.loc[cityA, cityB]
            if total_migration == 0:
                continue
            # weight = round(migration_value, 2) / total_migration
            weight = migration_value
            angle = Migration_angle.loc[cityA, cityB]
            direction_vector = np.array(angle_to_unit_vector(angle))
            total_vector += weight * direction_vector

        magnitude = np.linalg.norm(total_vector)/100
        unit_vector = total_vector / magnitude if magnitude != 0 else np.array([0.0, 0.0])


        total_vectors.append(total_vector)
        unit_vectors.append(unit_vector)
        vector_magnitudes.append(magnitude)
        city_name.append(infor.loc[cityA, 'city'])

    return unit_vectors, total_vectors, vector_magnitudes,city_name


if __name__ == '__main__':
    unit_vectors, total_vectors, vector_magnitudes,city_name=Global_vector(Migration_file='../data/Mobility21_23.csv')
    df['city_name'] = city_name
    df['global_vector'] = total_vectors
    df['global_vector_nom'] = unit_vectors
    df['vector_mag'] = vector_magnitudes
    df.to_csv('./out_average/Flow_vector_origin.csv',encoding='gbk')
    # df.to_csv('./out_clean_189/Flow_vector_origin.csv',encoding='gbk')