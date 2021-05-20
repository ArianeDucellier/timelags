"""
Script to prepare files for GMT
"""
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']
#arrays = ['BH', 'BS', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

# To compute distance along strike
lat0_1 = 47.9
lat0_2 = 48.0
lat0_3 = 48.1
lon0 = -123.125
a = 6378.136
e = 0.006694470
dx_1 = (pi / 180.0) * a * cos(lat0_1 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0_1 * pi / 180.0) * sin(lat0_1 * pi / 180.0))
dx_2 = (pi / 180.0) * a * cos(lat0_2 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0_2 * pi / 180.0) * sin(lat0_2 * pi / 180.0))
dx_3 = (pi / 180.0) * a * cos(lat0_3 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0_3 * pi / 180.0) * sin(lat0_3 * pi / 180.0))
dy_1 = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0_1 * \
    pi / 180.0) * sin(lat0_1 * pi / 180.0)) ** 1.5)
dy_2 = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0_2 * \
    pi / 180.0) * sin(lat0_2 * pi / 180.0)) ** 1.5)
dy_3 = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0_3 * \
    pi / 180.0) * sin(lat0_3 * pi / 180.0)) ** 1.5)
m_1 = 0.2 * dy_1 / dx_1
m_2 = 0.2 * dy_2 / dx_2
m_3 = 0.2 * dy_3 / dx_3

# Read output files
for num, array in enumerate(arrays):
    df = pickle.load(open('cc/{}/{}_{}_{}_width.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df = df.merge(quality, on=['i', 'j'], how='left', indicator=True)

    # Compute distance along strike
    strike_1 = np.zeros(len(df))
    strike_2 = np.zeros(len(df))
    strike_3 = np.zeros(len(df))
    for i in range(0, len(df)):
        latitude = df['latitude'][i]
        longitude = df['longitude'][i]
        x0 = (longitude - lon0) * dx_1
        y0 = (latitude - lat0_1) * dy_1
        x1 = (x0 + m_1 * y0) / (1 + m_1 ** 2.0)
        y1 = m_1 * x1
        distance = np.sign(y0 - m_1 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
        strike_1[i] = distance
        x0 = (longitude - lon0) * dx_2
        y0 = (latitude - lat0_2) * dy_2
        x1 = (x0 + m_2 * y0) / (1 + m_2 ** 2.0)
        y1 = m_2 * x1
        distance = np.sign(y0 - m_2 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
        strike_2[i] = distance
        x0 = (longitude - lon0) * dx_3
        y0 = (latitude - lat0_3) * dy_3
        x1 = (x0 + m_3 * y0) / (1 + m_3 ** 2.0)
        y1 = m_3 * x1
        distance = np.sign(y0 - m_3 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
        strike_3[i] = distance
    df['strike_1'] = strike_1
    df['strike_2'] = strike_2
    df['strike_3'] = strike_3

    # Drop values for which peak is too small
    df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop values with bad quality
    df.drop(df[df.quality > 2].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialization
    uncertainty = np.zeros((len(df), 3))
    depth = np.zeros((len(df), 3))
    d_to_pb_M = np.zeros((len(df), 3))
    d_to_pb_P = np.zeros((len(df), 3))
    section_uncertainty = np.zeros((len(df), 3))
    section_peak = np.zeros((len(df), 3))
    section_strike_1 = np.zeros((len(df), 3))
    section_strike_2 = np.zeros((len(df), 3))
    section_strike_3 = np.zeros((len(df), 3))

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
        section_strike_1[i, 0] = df['longitude'][i]
        section_strike_2[i, 0] = df['longitude'][i]
        section_strike_3[i, 0] = df['longitude'][i]
        if df['maxE'][i] > df['maxN'][i]:
            uncertainty[i, 2] = df['thick_EW'][i]
            depth[i, 2] = df['dist_EW'][i]
            d_to_pb_M[i, 2] = df['d_to_pb_EW_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_EW_P'][i]
            section_uncertainty[i, 1] = - df['dist_EW'][i]
            section_peak[i, 1] = - df['dist_EW'][i]
            section_strike_1[i, 1] = - df['dist_EW'][i]
            section_strike_2[i, 1] = - df['dist_EW'][i]
            section_strike_3[i, 1] = - df['dist_EW'][i]
            section_uncertainty[i, 2] = df['thick_EW'][i]
            section_peak[i, 2] = df['maxE'][i]
            section_strike_1[i, 2] = df['strike_1'][i]
            section_strike_2[i, 2] = df['strike_2'][i]
            section_strike_3[i, 2] = df['strike_3'][i]
        else:
            uncertainty[i, 2] = df['thick_NS'][i]
            depth[i, 2] = df['dist_NS'][i]
            d_to_pb_M[i, 2] = df['d_to_pb_NS_M'][i]
            d_to_pb_P[i, 2] = df['d_to_pb_NS_P'][i]
            section_uncertainty[i, 1] = - df['dist_NS'][i]
            section_peak[i, 1] = - df['dist_NS'][i]
            section_strike_1[i, 1] = - df['dist_NS'][i]
            section_strike_2[i, 1] = - df['dist_NS'][i]
            section_strike_3[i, 1] = - df['dist_NS'][i]
            section_uncertainty[i, 2] = df['thick_NS'][i]
            section_peak[i, 2] = df['maxN'][i]
            section_strike_1[i, 2] = df['strike_1'][i]
            section_strike_2[i, 2] = df['strike_2'][i]
            section_strike_3[i, 2] = df['strike_3'][i]

    np.savetxt('map_depth/uncertainty_{}_{}_{}.txt'.format(type_stack, cc_stack, array), uncertainty, fmt='%10.5f')
    np.savetxt('map_depth/depth_{}_{}_{}.txt'.format(type_stack, cc_stack, array), depth, fmt='%10.5f')
    np.savetxt('map_depth/d_to_pb_{}_{}_{}_M.txt'.format(type_stack, cc_stack, array), d_to_pb_M, fmt='%10.5f')
    np.savetxt('map_depth/d_to_pb_{}_{}_{}_P.txt'.format(type_stack, cc_stack, array), d_to_pb_P, fmt='%10.5f')
    np.savetxt('map_depth/section_uncertainty_{}_{}_{}.txt'.format(type_stack, cc_stack, array), section_uncertainty, fmt='%10.5f')
    np.savetxt('map_depth/section_peak_{}_{}_{}.txt'.format(type_stack, cc_stack, array), section_peak, fmt='%10.5f')

    section_strike_1 = section_strike_1[np.abs(section_strike_1[:, 2]) <= 10, :]
    section_strike_2 = section_strike_2[np.abs(section_strike_2[:, 2]) <= 10, :]
    section_strike_3 = section_strike_3[np.abs(section_strike_3[:, 2]) <= 10, :]

    np.savetxt('map_depth/section_strike_{}_{}_{}_1.txt'.format(type_stack, cc_stack, array), section_strike_1, fmt='%10.5f')
    np.savetxt('map_depth/section_strike_{}_{}_{}_2.txt'.format(type_stack, cc_stack, array), section_strike_2, fmt='%10.5f')
    np.savetxt('map_depth/section_strike_{}_{}_{}_3.txt'.format(type_stack, cc_stack, array), section_strike_3, fmt='%10.5f')

#df = pickle.load(open('../data/depth/McCrory/LFEs_Sweet_2014.pkl', 'rb'))

#depth = np.zeros((len(df), 3))
#section_1 = np.zeros((len(df), 3))
#section_2 = np.zeros((len(df), 3))
#section_3 = np.zeros((len(df), 3))
#d_to_pb = np.zeros((len(df), 3))

#for i in range(0, len(df)):
#    depth[i, 0] = df['longitude'][i]
#    depth[i, 1] = df['latitude'][i]
#    depth[i, 2] = df['depth'][i]
#    section_1[i, 0] = df['longitude'][i]
#    section_2[i, 0] = df['longitude'][i]
#    section_3[i, 0] = df['longitude'][i]
#    section_1[i, 1] = - df['depth'][i]
#    section_2[i, 1] = - df['depth'][i]
#    section_3[i, 1] = - df['depth'][i]
#    d_to_pb[i, 0] = df['longitude'][i]
#    d_to_pb[i, 1] = df['latitude'][i]
#    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

#    latitude = df['latitude'][i]
#    longitude = df['longitude'][i]
#    x0 = (longitude - lon0) * dx_1
#    y0 = (latitude - lat0_1) * dy_1
#    x1 = (x0 + m_1 * y0) / (1 + m_1 ** 2.0)
#    y1 = m_1 * x1
#    distance = np.sign(y0 - m_1 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_1[i, 2] = distance
#    x0 = (longitude - lon0) * dx_2
#    y0 = (latitude - lat0_2) * dy_2
#    x1 = (x0 + m_2 * y0) / (1 + m_2 ** 2.0)
#    y1 = m_2 * x1
#    distance = np.sign(y0 - m_2 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_2[i, 2] = distance
#    x0 = (longitude - lon0) * dx_3
#    y0 = (latitude - lat0_3) * dy_3
#    x1 = (x0 + m_3 * y0) / (1 + m_3 ** 2.0)
#    y1 = m_3 * x1
#    distance = np.sign(y0 - m_3 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_3[i, 2] = distance

#section_1 = section_1[np.abs(section_1[:, 2]) <= 10, :]
#section_2 = section_2[np.abs(section_2[:, 2]) <= 10, :]
#section_3 = section_3[np.abs(section_3[:, 2]) <= 10, :]

#np.savetxt('map_depth/depth_sweet.txt', depth, fmt='%10.5f')
#np.savetxt('map_depth/section_sweet_1.txt', section_1, fmt='%10.5f')
#np.savetxt('map_depth/section_sweet_2.txt', section_2, fmt='%10.5f')
#np.savetxt('map_depth/section_sweet_3.txt', section_3, fmt='%10.5f')
#np.savetxt('map_depth/d_to_pb_sweet_M.txt', d_to_pb, fmt='%10.5f')

#df = pickle.load(open('../data/depth/Preston/LFEs_Sweet_2014.pkl', 'rb'))

#d_to_pb = np.zeros((len(df), 3))

#for i in range(0, len(df)):
#    d_to_pb[i, 0] = df['longitude'][i]
#    d_to_pb[i, 1] = df['latitude'][i]
#    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

#np.savetxt('map_depth/d_to_pb_sweet_P.txt', d_to_pb, fmt='%10.5f')

#df = pickle.load(open('../data/depth/McCrory/LFEs_Chestler_2017.pkl', 'rb'))

#depth = np.zeros((len(df), 3))
#section_1 = np.zeros((len(df), 3))
#section_2 = np.zeros((len(df), 3))
#section_3 = np.zeros((len(df), 3))
#d_to_pb = np.zeros((len(df), 3))

#for i in range(0, len(df)):
#    depth[i, 0] = df['longitude'][i]
#    depth[i, 1] = df['latitude'][i]
#    depth[i, 2] = df['depth'][i]
#    section_1[i, 0] = df['longitude'][i]
#    section_2[i, 0] = df['longitude'][i]
#    section_3[i, 0] = df['longitude'][i]
#    section_1[i, 1] = - df['depth'][i]
#    section_2[i, 1] = - df['depth'][i]
#    section_3[i, 1] = - df['depth'][i]
#    d_to_pb[i, 0] = df['longitude'][i]
#    d_to_pb[i, 1] = df['latitude'][i]
#    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

#    latitude = df['latitude'][i]
#    longitude = df['longitude'][i]
#    x0 = (longitude - lon0) * dx_1
#    y0 = (latitude - lat0_1) * dy_1
#    x1 = (x0 + m_1 * y0) / (1 + m_1 ** 2.0)
#    y1 = m_1 * x1
#    distance = np.sign(y0 - m_1 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_1[i, 2] = distance
#    x0 = (longitude - lon0) * dx_2
#    y0 = (latitude - lat0_2) * dy_2
#    x1 = (x0 + m_2 * y0) / (1 + m_2 ** 2.0)
#    y1 = m_2 * x1
#    distance = np.sign(y0 - m_2 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_2[i, 2] = distance
#    x0 = (longitude - lon0) * dx_3
#    y0 = (latitude - lat0_3) * dy_3
#    x1 = (x0 + m_3 * y0) / (1 + m_3 ** 2.0)
#    y1 = m_3 * x1
#    distance = np.sign(y0 - m_3 * x0) * sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)
#    section_3[i, 2] = distance

#section_1 = section_1[np.abs(section_1[:, 2]) <= 10, :]
#section_2 = section_2[np.abs(section_2[:, 2]) <= 10, :]
#section_3 = section_3[np.abs(section_3[:, 2]) <= 10, :]

#np.savetxt('map_depth/depth_chestler.txt', depth, fmt='%10.5f')
#np.savetxt('map_depth/section_chestler_1.txt', section_1, fmt='%10.5f')
#np.savetxt('map_depth/section_chestler_2.txt', section_2, fmt='%10.5f')
#np.savetxt('map_depth/section_chestler_3.txt', section_3, fmt='%10.5f')
#np.savetxt('map_depth/d_to_pb_chestler_M.txt', d_to_pb, fmt='%10.5f')

#df = pickle.load(open('../data/depth/Preston/LFEs_Chestler_2017.pkl', 'rb'))

#d_to_pb = np.zeros((len(df), 3))

#for i in range(0, len(df)):
#    d_to_pb[i, 0] = df['longitude'][i]
#    d_to_pb[i, 1] = df['latitude'][i]
#    d_to_pb[i, 2] = df['depth'][i] - df['depth_pb'][i]

#np.savetxt('map_depth/d_to_pb_chestler_P.txt', d_to_pb, fmt='%10.5f')
