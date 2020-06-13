"""
Script to plot the difference in time lags bewteen
EW and NS components
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

lats = np.array([48.0056818181818, \
                    47.95728, \
                    48.068735, \
                    48.0059272727273, \
                    47.9321857142857, \
                    48.0554071428571, \
                    48.0549384615385, \
                    47.9730357142857])

lons = np.array([-123.084354545455, \
                    -122.92866, \
                    -122.969935, \
                    -123.313118181818, \
                    -123.045528571429, \
                    -123.210035714286, \
                    -123.464415384615, \
                    -123.138492857143])

type_stack = 'PWS'
cc_stack = 'PWS'

# Earth's radius and ellipticity
a = 6378.136
e = 0.006694470

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

plt.figure(1, figsize=(15, 15))

for num, (array, lat, lon) in enumerate(zip(arrays, lats, lons)):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_diff_EW_NS.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    dx = (pi / 180.0) * a * cos(lat * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat * pi / 180.0) * sin(lat * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat * \
        pi / 180.0) * sin(lat * pi / 180.0)) ** 1.5)
    x = (df_temp['longitude'] - lon) * dx
    y = (df_temp['latitude'] - lat) * dy
    az = 90.0 - (180.0 / pi) * np.arctan2(y, x)
    az = np.where(az < 0.0, 360.0 + az, az)
    df_temp['azimut'] = pd.Series(az, index=df_temp.index)
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

ax1 = plt.subplot(321)
plt.hist(df['diff_time'], \
    bins=[-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, \
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
plt.xlabel('Time difference (s)', fontsize=24)
plt.ylabel('Number of grid points', fontsize=24)
plt.title('EW and NS time lags', fontsize=30)

ax2 = plt.subplot(322)
plt.hist(df['diff_depth'], \
    bins=[-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, \
    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
plt.xlabel('Depth difference (km)', fontsize=24)
plt.ylabel('Number of grid points', fontsize=24)
plt.title('EW and NS depth difference', fontsize=30)

ax3 = plt.subplot(323)
plt.plot(df['distance'], df['diff_time'], 'ko')
plt.xlabel('Distance tremor - array (km)', fontsize=24)
plt.ylabel('Time difference (s)', fontsize=24)

ax4 = plt.subplot(324)
plt.plot(df['distance'], df['diff_depth'], 'ko')
plt.xlabel('Distance tremor - array (km)', fontsize=24)
plt.ylabel('Depth difference (km)', fontsize=24)

ax5 = plt.subplot(325)
plt.plot(df['azimut'], df['diff_time'], 'ko')
plt.xlabel('Azimuth', fontsize=24)
plt.ylabel('Time difference (s)', fontsize=24)

ax6 = plt.subplot(326)
plt.plot(df['azimut'], df['diff_depth'], 'ko')
plt.xlabel('Azimuth', fontsize=24)
plt.ylabel('Depth difference (km)', fontsize=24)

plt.tight_layout()
plt.savefig('diff_EW_NS/{}_{}.eps'.format(type_stack, cc_stack), format='eps')
ax1.clear()
ax2.clear()
ax3.clear()
ax4.clear()
ax5.clear()
ax6.clear()
plt.close(1)
