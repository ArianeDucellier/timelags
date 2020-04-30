"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'lin'
cc_stack = 'lin'

for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_thick.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)
df = df.dropna()

STD = np.zeros((len(df), 3))
MAD = np.zeros((len(df), 3))
S = np.zeros((len(df), 3))
Q = np.zeros((len(df), 3))

for i in range(0, len(df)):
    # Thickness from standard deviation
    STD[i, 0] = df['longitude'].iloc[i]
    STD[i, 1] = df['latitude'].iloc[i]
    if (df['ratioE'].iloc[i] > df['ratioN'].iloc[i]):
        STD[i, 2] = df['STD_thick_EW'].iloc[i]
    else:
        STD[i, 2] = df['STD_thick_NS'].iloc[i]
    # Thickness from MAD estimator
    MAD[i, 0] = df['longitude'].iloc[i]
    MAD[i, 1] = df['latitude'].iloc[i]
    if (df['ratioE'].iloc[i] > df['ratioN'].iloc[i]):
        MAD[i, 2] = df['MAD_thick_EW'].iloc[i]
    else:
        MAD[i, 2] = df['MAD_thick_NS'].iloc[i]
    # Thickness from S estimator
    S[i, 0] = df['longitude'].iloc[i]
    S[i, 1] = df['latitude'].iloc[i]
    if (df['ratioE'].iloc[i] > df['ratioN'].iloc[i]):
        S[i, 2] = df['S_thick_EW'].iloc[i]
    else:
        S[i, 2] = df['S_thick_NS'].iloc[i]
    # Thickness from Q estimator
    Q[i, 0] = df['longitude'].iloc[i]
    Q[i, 1] = df['latitude'].iloc[i]
    if (df['ratioE'].iloc[i] > df['ratioN'].iloc[i]):
        Q[i, 2] = df['Q_thick_EW'].iloc[i]
    else:
        Q[i, 2] = df['Q_thick_NS'].iloc[i]

np.savetxt('map_thick/STD_{}_{}.txt'.format(type_stack, cc_stack), STD)
np.savetxt('map_thick/MAD_{}_{}.txt'.format(type_stack, cc_stack), MAD)
np.savetxt('map_thick/S_{}_{}.txt'.format(type_stack, cc_stack), S)
np.savetxt('map_thick/Q_{}_{}.txt'.format(type_stack, cc_stack), Q)
