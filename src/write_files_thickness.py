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
    df1 = pickle.load(open('cc/{}/{}_{}_{}_width_reloc.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df1 = df1.merge(quality, on=['i', 'j'], how='left', indicator=True)

    df2 = pickle.load(open('cc/{}/{}_{}_{}_thick_reloc.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))

    df = df1.merge(df2, on=['i', 'j', 'latitude', 'longitude', 'distance', \
        'ntremor', 'ratioE', 'ratioN'], how='left', indicator='merge2')

    # Drop values for which peak is too small
    df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop values with bad quality
    df.drop(df[df.quality > 2].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    STD = np.zeros((len(df), 3))
    MAD = np.zeros((len(df), 3))
    S = np.zeros((len(df), 3))
    Q = np.zeros((len(df), 3))

    for i in range(0, len(df)):
        STD[i, 0] = df['longitude'].iloc[i]
        STD[i, 1] = df['latitude'].iloc[i]
        MAD[i, 0] = df['longitude'].iloc[i]
        MAD[i, 1] = df['latitude'].iloc[i]
        S[i, 0] = df['longitude'].iloc[i]
        S[i, 1] = df['latitude'].iloc[i]
        Q[i, 0] = df['longitude'].iloc[i]
        Q[i, 1] = df['latitude'].iloc[i]
        if df['maxE'][i] > df['maxN'][i]:
            STD[i, 2] = df['STD_EW'].iloc[i]
            MAD[i, 2] = df['MAD_EW'].iloc[i]
            S[i, 2] = df['S_EW'].iloc[i]
            Q[i, 2] = df['Q_EW'].iloc[i]
        else:
            STD[i, 2] = df['STD_NS'].iloc[i]
            MAD[i, 2] = df['MAD_NS'].iloc[i]
            S[i, 2] = df['S_NS'].iloc[i]
            Q[i, 2] = df['Q_NS'].iloc[i]

    np.savetxt('map_thick/STD_{}_{}_{}_reloc.txt'.format(type_stack, cc_stack, array), STD, fmt='%10.5f')
    np.savetxt('map_thick/MAD_{}_{}_{}_reloc.txt'.format(type_stack, cc_stack, array), MAD, fmt='%10.5f')
    np.savetxt('map_thick/S_{}_{}_{}_reloc.txt'.format(type_stack, cc_stack, array), S, fmt='%10.5f')
    np.savetxt('map_thick/Q_{}_{}_{}_reloc.txt'.format(type_stack, cc_stack, array), Q, fmt='%10.5f')
