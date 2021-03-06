"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

for num, array in enumerate(arrays):
    df1_temp = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df1_temp = df1_temp.merge(quality, on=['i', 'j'], how='left', indicator=True)
    if (num == 0):
        df1 = df1_temp
    else:
        df1 = pd.concat([df1, df1_temp], ignore_index=True)

for num, array in enumerate(arrays):
    df2_temp = pickle.load(open('cc/{}/keep/{}_{}_{}_thick_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df2 = df2_temp
    else:
        df2 = pd.concat([df2, df2_temp], ignore_index=True)

df = df1.merge(df2, on=['i', 'j', 'latitude', 'longitude', 'distance', \
    'ntremor', 'ratioE', 'ratioN'], how='left', indicator='merge2')

# Drop values for which peak is too small
df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop values with bad quality
df.drop(df[df.quality != 1].index, inplace=True)
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

#STD = STD[STD[:, 2] > 0.0, :]
#MAD = MAD[MAD[:, 2] > 0.0, :]
#S = S[S[:, 2] > 0.0, :]
#Q = Q[Q[:, 2] > 0.0, :]

np.savetxt('map_thick/STD_{}_{}.txt'.format(type_stack, cc_stack), STD, fmt='%10.5f')
np.savetxt('map_thick/MAD_{}_{}.txt'.format(type_stack, cc_stack), MAD, fmt='%10.5f')
np.savetxt('map_thick/S_{}_{}.txt'.format(type_stack, cc_stack), S, fmt='%10.5f')
np.savetxt('map_thick/Q_{}_{}.txt'.format(type_stack, cc_stack), Q, fmt='%10.5f')
