"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

arrays = ['BH', 'BS', 'DR', 'GC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

# To compute distance along strike
lat0 = 48.0
lon0 = -123.125
a = 6378.136
e = 0.006694470
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
    pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
m = 0.2 * dy / dx

# Read output files
for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

# Compute distance along strike
strike = np.zeros(len(df))
for i in range(0, len(df)):
    latitude = df['latitude'][i]
    longitude = df['longitude'][i]
    x0 = (longitude - lon0) * dx
    y0 = (latitude - lat0) * dy
    x1 = (x0 + m * y0) / (1 + m ** 2.0)
    y1 = m * x1
    distance = sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
    strike[i] = distance
df['strike'] = strike

# Drop values for which peak is too small
df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Initialization
uncertainty = np.zeros((len(df), 3))
depth = np.zeros((len(df), 3))
d_to_pb_M = np.zeros((len(df), 3))
d_to_pb_P = np.zeros((len(df), 3))
section_uncertainty = np.zeros((len(df), 3))
section_peak = np.zeros((len(df), 3))
section_strike = np.zeros((len(df), 3))

# Fill values with component with highest peak
for i in range(0, len(df)):
    uncertainty[i, 0] = df['longitude'][i]
    uncertainty[i, 1] = df['latitude'][i]
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    d_to_pb_M[i, 0] = df['longitude'][i]
    d_to_pb_M[i, 1] = df['latitude'][i]
    d_to_pb_P[i, 0] = df['longitude'][i]
    d_to_pb_P[i, 1] = df['latitude'][i]
    section_uncertainty[i, 0] = df['longitude'][i]
    section_peak[i, 0] = df['longitude'][i]
    section_strike[i, 0] = df['longitude'][i]
    if df['maxE'][i] > df['maxN'][i]:
        uncertainty[i, 2] = df['thick_EW'][i]
        depth[i, 2] = df['dist_EW'][i]
        d_to_pb_M[i, 2] = df['d_to_pb_EW_M'][i]
        d_to_pb_P[i, 2] = df['d_to_pb_EW_P'][i]
        section_uncertainty[i, 1] = - df['dist_EW'][i]
        section_peak[i, 1] = - df['dist_EW'][i]
        section_strike[i, 1] = - df['dist_EW'][i]
        section_uncertainty[i, 2] = df['thick_EW'][i]
        section_peak[i, 2] = df['maxE'][i]
        section_strike[i, 2] = df['strike'][i]
    else:
        uncertainty[i, 2] = df['thick_NS'][i]
        depth[i, 2] = df['dist_NS'][i]
        d_to_pb_M[i, 2] = df['d_to_pb_NS_M'][i]
        d_to_pb_P[i, 2] = df['d_to_pb_NS_P'][i]
        section_uncertainty[i, 1] = - df['dist_NS'][i]
        section_peak[i, 1] = - df['dist_NS'][i]
        section_strike[i, 1] = - df['dist_NS'][i]
        section_uncertainty[i, 2] = df['thick_NS'][i]
        section_peak[i, 2] = df['maxN'][i]
        section_strike[i, 2] = df['strike'][i]

np.savetxt('map_depth/uncertainty_{}_{}.txt'.format(type_stack, cc_stack), uncertainty, fmt='%10.5f')
np.savetxt('map_depth/depth_{}_{}.txt'.format(type_stack, cc_stack), depth, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_{}_{}_M.txt'.format(type_stack, cc_stack), d_to_pb_M, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_{}_{}_P.txt'.format(type_stack, cc_stack), d_to_pb_P, fmt='%10.5f')
np.savetxt('map_depth/section_uncertainty_{}_{}.txt'.format(type_stack, cc_stack), section_uncertainty, fmt='%10.5f')
np.savetxt('map_depth/section_peak_{}_{}.txt'.format(type_stack, cc_stack), section_peak, fmt='%10.5f')
np.savetxt('map_depth/section_strike_{}_{}.txt'.format(type_stack, cc_stack), section_strike, fmt='%10.5f')

df = pickle.load(open('../data/depth/McCrory/LFEs_Sweet_2014.pkl', 'rb'))

depth = np.zeros((len(df), 3))
section = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    section[i, 0] = df['longitude'][i]
    section[i, 1] = - df['depth'][i]
    section[i, 2] = df['depth'][i]
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/depth_sweet.txt', depth, fmt='%10.5f')
np.savetxt('map_depth/section_sweet.txt', section, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_sweet_M.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/Preston/LFEs_Sweet_2014.pkl', 'rb'))

d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/d_to_pb_sweet_P.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/McCrory/LFEs_Chestler_2017.pkl', 'rb'))

depth = np.zeros((len(df), 3))
section = np.zeros((len(df), 3))
d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    depth[i, 0] = df['longitude'][i]
    depth[i, 1] = df['latitude'][i]
    depth[i, 2] = df['depth'][i]
    section[i, 0] = df['longitude'][i]
    section[i, 1] = - df['depth'][i]
    section[i, 2] = df['depth'][i]
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/depth_chestler.txt', depth, fmt='%10.5f')
np.savetxt('map_depth/section_chestler.txt', section, fmt='%10.5f')
np.savetxt('map_depth/d_to_pb_chestler_M.txt', d_to_pb, fmt='%10.5f')

df = pickle.load(open('../data/depth/Preston/LFEs_Chestler_2017.pkl', 'rb'))

d_to_pb = np.zeros((len(df), 3))

for i in range(0, len(df)):
    d_to_pb[i, 0] = df['longitude'][i]
    d_to_pb[i, 1] = df['latitude'][i]
    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

np.savetxt('map_depth/d_to_pb_chestler_P.txt', d_to_pb, fmt='%10.5f')
