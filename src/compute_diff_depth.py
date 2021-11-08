"""
Script to compute the difference in depth corresponding
to different sizes of the time window where we do the clustering
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

diff_12 = np.array([])
diff_32 = np.array([])

for num, array in enumerate(arrays):
    df_lag1 = pickle.load(open('cc/{}/{}_{}_{}_width_tlag1.pkl'.format(array, array, type_stack, cc_stack), 'rb'))
    df_lag2 = pickle.load(open('cc/{}/{}_{}_{}_width_tlag2.pkl'.format(array, array, type_stack, cc_stack), 'rb'))
    df_lag3 = pickle.load(open('cc/{}/{}_{}_{}_width_tlag3.pkl'.format(array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format(array, type_stack, cc_stack), 'rb'))

    df_lag1 = df_lag1.merge(quality, on=['i', 'j'], how='left', indicator=True)
    df_lag2 = df_lag2.merge(quality, on=['i', 'j'], how='left', indicator=True)
    df_lag3 = df_lag3.merge(quality, on=['i', 'j'], how='left', indicator=True)

    df_lag1.drop(df_lag1[(df_lag1.maxE < threshold) & (df_lag1.maxN < threshold)].index, inplace=True)
    df_lag1.reset_index(drop=True, inplace=True)
    df_lag2.drop(df_lag2[(df_lag2.maxE < threshold) & (df_lag2.maxN < threshold)].index, inplace=True)
    df_lag2.reset_index(drop=True, inplace=True)
    df_lag3.drop(df_lag3[(df_lag3.maxE < threshold) & (df_lag3.maxN < threshold)].index, inplace=True)
    df_lag3.reset_index(drop=True, inplace=True)

    df_lag1.drop(df_lag1[df_lag1.quality > 2].index, inplace=True)
    df_lag1.reset_index(drop=True, inplace=True)
    df_lag2.drop(df_lag2[df_lag2.quality > 2].index, inplace=True)
    df_lag2.reset_index(drop=True, inplace=True)
    df_lag3.drop(df_lag3[df_lag3.quality > 2].index, inplace=True)
    df_lag3.reset_index(drop=True, inplace=True)

    depth_lag1 = np.zeros(len(df_lag1))
    depth_lag2 = np.zeros(len(df_lag2))
    depth_lag3 = np.zeros(len(df_lag3))
    
    for i in range(0, len(df_lag1)):
        if df_lag1['maxE'][i] > df_lag1['maxN'][i]:
            depth_lag1[i] = df_lag1['dist_EW'][i]
        else:
            depth_lag1[i] = df_lag1['dist_NS'][i]

    for i in range(0, len(df_lag2)):
        if df_lag2['maxE'][i] > df_lag2['maxN'][i]:
            depth_lag2[i] = df_lag2['dist_EW'][i]
        else:
            depth_lag2[i] = df_lag2['dist_NS'][i]

    for i in range(0, len(df_lag3)):
        if df_lag3['maxE'][i] > df_lag3['maxN'][i]:
            depth_lag3[i] = df_lag3['dist_EW'][i]
        else:
            depth_lag3[i] = df_lag3['dist_NS'][i]

    diff_12 = np.concatenate([diff_12, depth_lag1 - depth_lag2])
    diff_32 = np.concatenate([diff_32, depth_lag3 - depth_lag2])

    diff_12 = diff_12[np.where(~np.isnan(diff_12))[0]]
    diff_32 = diff_32[np.where(~np.isnan(diff_32))[0]]

plt.figure(1)
plt.subplot2grid((1, 2), (0, 0))
plt.hist(diff_12)
plt.xlabel('Depth difference (km)')
plt.title('Time window of 2s versus 4s')
plt.subplot2grid((1, 2), (0, 1))
plt.hist(diff_32)
plt.xlabel('Depth difference (km)')
plt.title('Time window of 6s versus 4s')
plt.savefig('diff_depth/hist.eps', format='eps')

np.savetxt('diff_depth/diff_12.txt', diff_12, fmt='%10.5f')
np.savetxt('diff_depth/diff_32.txt', diff_32, fmt='%10.5f')
    