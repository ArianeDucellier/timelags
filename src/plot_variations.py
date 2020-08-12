# Script to plot variations of depth with Vp/Vs ratio

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'DR', 'GC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

# Read output files
for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df0 = df_temp
    else:
        df0 = pd.concat([df0, df_temp], ignore_index=True)

for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_m1.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df1 = df_temp
    else:
        df1 = pd.concat([df1, df_temp], ignore_index=True)

for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_p1.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df2 = df_temp
    else:
        df2 = pd.concat([df2, df_temp], ignore_index=True)

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

plt.figure(1, figsize=(12, 6))

df = df0.merge(df1, on=['i', 'j', 'latitude', 'longitude', 'distance', \
    'ntremor', 'ratioE', 'ratioN', 'maxE', 'maxN'], how='left', indicator=True)

# Drop values for which peak is too small
df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

distance = np.zeros(len(df))
time = np.zeros(len(df))
variations = np.zeros(len(df))
for i in range(0, len(df)):
    distance[i] = df['distance'][i]
    if df['maxE'][i] > df['maxN'][i]:
        time[i] = df['time_EW_x'][i]
        variations[i] = df['dist_EW_y'][i] - df['dist_EW_x'][i]
    else:
        time[i] = df['time_NS_x'][i]
        variations[i] = df['dist_NS_y'][i] - df['dist_NS_x'][i]

#ax1 = plt.subplot(221)
#plt.plot(time, variations, 'ko')
#plt.xlabel('Time (s)', fontsize=20)
#plt.ylabel('Depth difference (km)', fontsize=20)
#plt.title('Smaller Vp / Vs', fontsize=20)

ax1 = plt.subplot(121)
plt.plot(distance, variations, 'ko')
plt.xlabel('Distance from array (km)', fontsize=20)
plt.ylabel('Depth difference (km)', fontsize=20)
plt.title('Smaller Vp / Vs', fontsize=20)

df = df0.merge(df2, on=['i', 'j', 'latitude', 'longitude', 'distance', \
    'ntremor', 'ratioE', 'ratioN', 'maxE', 'maxN'], how='left', indicator=True)

# Drop values for which peak is too small
df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

distance = np.zeros(len(df))
time = np.zeros(len(df))
variations = np.zeros(len(df))
for i in range(0, len(df)):
    distance[i] = df['distance'][i]
    if df['maxE'][i] > df['maxN'][i]:
        time[i] = df['time_EW_x'][i]
        variations[i] = df['dist_EW_y'][i] - df['dist_EW_x'][i]
    else:
        time[i] = df['time_NS_x'][i]
        variations[i] = df['dist_NS_y'][i] - df['dist_NS_x'][i]

#ax3 = plt.subplot(222)
#plt.plot(time, variations, 'ko')
#plt.xlabel('Time (s)', fontsize=20)
#plt.ylabel('Depth difference (km)', fontsize=20)
#plt.title('Larger Vp / Vs', fontsize=20)

ax2 = plt.subplot(122)
plt.plot(distance, variations, 'ko')
plt.xlabel('Distance from array (km)', fontsize=20)
plt.ylabel('Depth difference (km)', fontsize=20)
plt.title('Larger Vp / Vs', fontsize=20)

plt.tight_layout()
plt.savefig('variations/{}_{}.eps'.format(type_stack, cc_stack), format='eps')
ax1.clear()
ax2.clear()
#ax3.clear()
#ax4.clear()
plt.close(1)