"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'GC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 4.0

for num, array in enumerate(arrays):
    df = pickle.load(open('cc/{}/{}_{}_{}_width_p1.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    df.drop(df[(df.thick_EW < 0.01) | (df.thick_NS < 0.01)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(df[(df.thick_EW > threshold) & (df.thick_NS > threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    d_to_pb_M = np.zeros((len(df), 4))
    d_to_pb_P = np.zeros((len(df), 4))

    for i in range(0, len(df)):
        d_to_pb_M[i, 0] = df['longitude'][i]
        d_to_pb_M[i, 1] = df['latitude'][i]
        d_to_pb_P[i, 0] = df['longitude'][i]
        d_to_pb_P[i, 1] = df['latitude'][i]
        if df['thick_EW'][i] < df['thick_NS'][i]:
            d_to_pb_M[i, 2] = df['d_to_pb_EW_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_EW_P'][i]
        else:
            d_to_pb_M[i, 2] = df['d_to_pb_NS_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_NS_P'][i]

    d_to_pb_M[:, 3] = 0.1 * d_to_pb_M[:, 2]
    d_to_pb_P[:, 3] = 0.1 * d_to_pb_P[:, 2]

    np.savetxt('map_depth/d_to_pb_{}_{}_{}_M_p1.txt'.format(type_stack, cc_stack, array), d_to_pb_M, fmt='%10.5f')
    np.savetxt('map_depth/d_to_pb_{}_{}_{}_P_p1.txt'.format(type_stack, cc_stack, array), d_to_pb_P, fmt='%10.5f')
