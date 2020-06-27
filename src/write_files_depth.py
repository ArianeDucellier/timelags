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
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

df.drop(df[(df.thick_EW < 0.01) | (df.thick_NS < 0.01)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

uncertainty = np.zeros((len(df), 3))

for i in range(0, len(df)):
    uncertainty[i, 0] = df['longitude'][i]
    uncertainty[i, 1] = df['latitude'][i]
    if df['thick_EW'][i] < df['thick_NS'][i]:
        uncertainty[i, 2] = df['thick_EW'][i]
    else:
        uncertainty[i, 2] = df['thick_NS'][i]

df.drop(df[(df.thick_EW > threshold) & (df.thick_NS > threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

depth = np.zeros((len(df), 3))
d_to_pb_M = np.zeros((len(df), 4))
d_to_pb_P = np.zeros((len(df), 4))

for i in range(0, len(df)):
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    d_to_pb_M[i, 0] = df['longitude'][i]
    d_to_pb_M[i, 1] = df['latitude'][i]
    d_to_pb_P[i, 0] = df['longitude'][i]
    d_to_pb_P[i, 1] = df['latitude'][i]
    if df['thick_EW'][i] < df['thick_NS'][i]:
        depth[i, 2] = df['dist_EW'][i]
        d_to_pb_M[i, 2] = df['d_to_pb_EW_M'][i]
        d_to_pb_P[i, 2] = df['d_to_pb_EW_P'][i]
    else:
        depth[i, 2] = df['dist_NS'][i]
        d_to_pb_M[i, 2] = df['d_to_pb_NS_M'][i]
        d_to_pb_P[i, 2] = df['d_to_pb_NS_P'][i]

d_to_pb_M[:, 3] = 0.1 * d_to_pb_M[:, 2]
d_to_pb_P[:, 3] = 0.1 * d_to_pb_P[:, 2]

np.savetxt('map_depth/depth_{}_{}.txt'.format(type_stack, cc_stack), depth, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_{}_{}_M.txt'.format(type_stack, cc_stack), d_to_pb_M, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_{}_{}_P.txt'.format(type_stack, cc_stack), d_to_pb_P, fmt='%10.5f')
np.savetxt('map_depth/uncertainty_{}_{}.txt'.format(type_stack, cc_stack), uncertainty, fmt='%10.5f')

df = pickle.load(open('../data/depth/McCrory/LFEs_Sweet_2014.pkl', 'rb'))

depth = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 4))

for i in range(0, len(df)):
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

d_to_pb[:, 3] = 0.1 * d_to_pb[:, 2]
    
np.savetxt('map_depth/depth_sweet.txt', depth, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_sweet_M.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/Preston/LFEs_Sweet_2014.pkl', 'rb'))

d_to_pb = np.zeros((len(df), 4))

for i in range(0, len(df)):
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

d_to_pb[:, 3] = 0.1 * d_to_pb[:, 2]

np.savetxt('map_depth/d_to_pb_sweet_P.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/McCrory/LFEs_Chestler_2017.pkl', 'rb'))

depth = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 4))

for i in range(0, len(df)):
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

d_to_pb[:, 3] = 0.1 * d_to_pb[:, 2]

np.savetxt('map_depth/depth_chestler.txt', depth, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_chestler_M.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/Preston/LFEs_Chestler_2017.pkl', 'rb'))

d_to_pb = np.zeros((len(df), 4))

d_to_pb[:, 3] = 0.1 * d_to_pb[:, 2]

for i in range(0, len(df)):
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/d_to_pb_chestler_P.txt', d_to_pb, fmt='%10.5f')
