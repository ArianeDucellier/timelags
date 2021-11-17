"""
Script to write table for supplement
"""
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

latitudes = np.array([48.0056818181818, 47.95728, 48.068735, 48.0059272727273, \
    47.932185714285, 48.0554071428571, 48.0549384615385, 47.9730357142857])
longitudes = np.array([-123.084354545455, -122.92866, -122.969935, -123.313118181818, \
    -123.045528571429, -123.210035714286, -123.464415384615, -123.138492857143])

table = np.ndarray(shape=(0, 7))

for num, array in enumerate(arrays):
    df1 = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df1 = df1.merge(quality, on=['i', 'j'], how='left', indicator=True)

    df2 = pickle.load(open('cc/{}/{}_{}_{}_thick.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))

    df = df1.merge(df2, on=['i', 'j', 'latitude', 'longitude', 'distance', \
        'ntremor', 'ratioE', 'ratioN'], how='left', indicator='merge2')
    
    # Drop values for which peak is too small
    df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop values with bad quality
    df.drop(df[df.quality > 2].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialization
    table1 = np.zeros((len(df), 7))

    # Fill values with component with highest peak
    for i in range(0, len(df)):
        table1[i, 0] = df['latitude'][i]
        table1[i, 1] = df['longitude'][i]
        table1[i, 5] = latitudes[num]
        table1[i, 6] = longitudes[num]
        if df['maxE'][i] > df['maxN'][i]:
            table1[i, 2] = df['dist_EW'][i]
            table1[i, 3] = df['Q_EW'].iloc[i]
            table1[i, 4] = df['time_EW'].iloc[i]
        else:
            table1[i, 2] = df['dist_NS'][i]
            table1[i, 3] = df['Q_NS'].iloc[i]
            table1[i, 4] = df['time_NS'].iloc[i]
            
    table = np.concatenate([table, table1])

np.savetxt('table_supp.txt', table, fmt='%10.5f')
