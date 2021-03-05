"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

#arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']
arrays = ['BH', 'BS', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

for num, array in enumerate(arrays):
    df = pickle.load(open('cc/{}/{}_{}_{}_width_reloc.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df = df.merge(quality, on=['i', 'j'], how='left', indicator=True)
    df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(df[df.quality != 1].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    d_to_pb_M = np.zeros((len(df), 3))
    d_to_pb_P = np.zeros((len(df), 3))

    for i in range(0, len(df)):
        d_to_pb_M[i, 0] = df['longitude'][i]
        d_to_pb_M[i, 1] = df['latitude'][i]
        d_to_pb_P[i, 0] = df['longitude'][i]
        d_to_pb_P[i, 1] = df['latitude'][i]
        if df['maxE'][i] > df['maxN'][i]:
            d_to_pb_M[i, 2] = df['d_to_pb_EW_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_EW_P'][i]
        else:
            d_to_pb_M[i, 2] = df['d_to_pb_NS_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_NS_P'][i]

    np.savetxt('map_depth/d_to_pb_{}_{}_{}_M_reloc.txt'.format(type_stack, cc_stack, array), d_to_pb_M, fmt='%10.5f')
    np.savetxt('map_depth/d_to_pb_{}_{}_{}_P_reloc.txt'.format(type_stack, cc_stack, array), d_to_pb_P, fmt='%10.5f')
