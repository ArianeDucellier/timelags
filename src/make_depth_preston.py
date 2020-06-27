import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt
from scipy import interpolate

#arrayName = 'BH'
#lat0 = 48.0056818181818
#lon0 = -123.084354545455

#arrayName = 'BS'
#lat0 = 47.95728
#lon0 = -122.92866

#arrayName = 'CL'
#lat0 = 48.068735
#lon0 = -122.969935

#arrayName = 'DR'
#lat0 = 48.0059272727273
#lon0 = -123.313118181818

#arrayName = 'GC'
#lat0 = 47.9321857142857
#lon0 = -123.045528571429

arrayName = 'LC'
lat0 = 48.0554071428571
lon0 = -123.210035714286

#arrayName = 'PA'
#lat0 = 48.0549384615385
#lon0 = -123.464415384615

#arrayName = 'TB'
#lat0 = 47.9730357142857
#lon0 = -123.138492857143

ds = 5.0

a = 6378.136
e = 0.006694470
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
    pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

model = np.loadtxt('../matlab/slabTopContours_5km.txt')
f = interpolate.interp2d(model[: , 0], model[: , 1], model[: , 2], kind='linear')

depth = np.zeros((121, 3))

index = 0
for j in range(-5, 6):
    for i in range(-5, 6):
        x0 = i * ds
        y0 = j * ds
        longitude = lon0 + x0 / dx
        latitude = lat0 + y0 / dy
        my_depth = - f(longitude, latitude)
        depth[index, 0] = x0
        depth[index, 1] = y0
        depth[index, 2] = my_depth
        index = index + 1

#df = pickle.load(open('../data/depth/McCrory/LFEs_Chestler_2017.pkl', 'rb'))

#for i in range(0, len(df)):
#    my_depth = f(df['longitude'].iloc[i], df['latitude'].iloc[i])
#    df.at[i, 'depth_pb'] = my_depth


#pickle.dump(df, open('../data/depth/Preston/LFEs_Chestler_2017.pkl', 'wb'))

np.savetxt('../data/depth/Preston/' + arrayName + '_depth.txt', depth)
        