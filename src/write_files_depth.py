"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

depth = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 3))
uncertainty = np.zeros((len(df), 3))

for i in range(0, len(df)):
    # Depth of tremor
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    if (df['ratioE'][i] > df['ratioN'][i]):
        depth[i, 2] = df['dist_EW'][i]
    else:
        depth[i, 2] = df['dist_NS'][i]
    # Distance to plate boundary
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    if (df['ratioE'][i] > df['ratioN'][i]):
        d_to_pb[i, 2] = df['d_to_pb_EW'][i]
    else:
        d_to_pb[i, 2] = df['d_to_pb_NS'][i]
    # Uncertainty on depth
    uncertainty[i, 0] = df['longitude'][i]
    uncertainty[i, 1] = df['latitude'][i]
    if (df['ratioE'][i] > df['ratioN'][i]):
        uncertainty[i, 2] = df['thick_EW'][i]
    else:
        uncertainty[i, 2] = df['thick_NS'][i]

np.savetxt('map_depth/depth_{}_{}.txt'.format(type_stack, cc_stack), depth)
np.savetxt('map_depth/d_to_pb_{}_{}.txt'.format(type_stack, cc_stack), d_to_pb)
np.savetxt('map_depth/uncertainty_{}_{}.txt'.format(type_stack, cc_stack), uncertainty)

df = pickle.load(open('../data/depth/LFEs_Sweet_2014.pkl', 'rb'))

depth = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    # Depth of tremor
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    # Distance to plate boundary
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/depth_sweet.txt', depth)
np.savetxt('map_depth/d_to_pb_sweet.txt', d_to_pb)

df = pickle.load(open('../data/depth/LFEs_Chestler_2017.pkl', 'rb'))

depth = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    # Depth of tremor
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    # Distance to plate boundary
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/depth_chestler.txt', depth)
np.savetxt('map_depth/d_to_pb_chestler.txt', d_to_pb)
