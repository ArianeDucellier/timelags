"""
Script to compute the time lag between the direct P-wave
and the direct S-wave using cross-correlation of seismic
signal recorded during tectonic tremor
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import obspy
import os
import pandas as pd
import pickle

from math import sqrt

from cluster_select import cluster_select
from plot_stack_acorr import plot_stack_acorr
from plot_stack_ccorr import plot_stack_ccorr
from stacking import linstack, powstack, PWstack

from misc import get_travel_time

# Set parameters
#arrayNames = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']
arrayNames = ['TB']

h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
vp0 = np.array([4.8802, 5.0365, 5.3957, 6.2648, 6.8605, 6.7832, 6.4347, 6.1937, 6.0609, 6.029, 6.1562, 6.5261, 6.8914, 7.2201, 7.5902, 7.99, 7.8729, 7.9578, 8.0181, 8.0238, 8.0381])
vs0 = 1.00 * np.array([2.3612, 2.5971, 2.9289, 3.3382, 3.7488, 3.8076, 3.6282, 3.5566, 3.5032, 3.4848, 3.5262, 3.6438, 3.8624, 4.103, 4.316, 4.5194, 4.5373, 4.6046, 4.6349, 4.6381, 4.6461])

w = 2.0
Tmax = 15.0
ds = 5.0
ncor_cluster = 40
RMSmin = 12.0
RMSmax = 14.0
xmin = 2.0
xmax = 8.0
nc = 2
palette = {0: 'tomato', 1: 'royalblue', 2:'forestgreen', 3:'gold', \
    4: 'lightpink', 5:'skyblue'}
Vs = 3.6
Vp = 6.4

# Loop on arrays
for arrayName in arrayNames:

    # Get depth of plate boundary around the array
    depth = pd.read_csv('../data/depth/McCrory/' + arrayName + '_depth.txt', \
        sep=' ', header=None)
    depth.columns = ['x', 'y', 'depth']

    # Preston model
    depth_pb_P = pd.read_csv('../data/depth/Preston/' + arrayName + '_depth.txt', sep=' ', \
        header=None)
    depth_pb_P.columns = ['x', 'y', 'depth']

    # Final results
    df_width = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        arrayName, arrayName, 'PWS', 'PWS'), 'rb'))

    # Store results in pandas dataframe
    namefile = arrayName + '_timelag.pkl'
    if os.path.exists(namefile):
        df = pickle.load(open(namefile, 'rb'))
    else:
        df = pd.DataFrame(columns=['x0', 'y0', 'ntremor', \
            't_lin_lin_EW', 't_lin_pow_EW', 't_lin_PWS_EW', \
            't_lin_lin_NS', 't_lin_pow_NS', 't_lin_PWS_NS', \
            't_pow_lin_EW', 't_pow_pow_EW', 't_pow_PWS_EW', \
            't_pow_lin_NS', 't_pow_pow_NS', 't_pow_PWS_NS', \
            't_PWS_lin_EW', 't_PWS_pow_EW', 't_PWS_PWS_EW', \
            't_PWS_lin_NS', 't_PWS_pow_NS', 't_PWS_PWS_NS', \
            't_lin_lin_EW_cluster', 't_lin_pow_EW_cluster', \
            't_lin_PWS_EW_cluster', 't_lin_lin_NS_cluster', \
            't_lin_pow_NS_cluster', 't_lin_PWS_NS_cluster', \
            't_pow_lin_EW_cluster', 't_pow_pow_EW_cluster', \
            't_pow_PWS_EW_cluster', 't_pow_lin_NS_cluster', \
            't_pow_pow_NS_cluster', 't_pow_PWS_NS_cluster', \
            't_PWS_lin_EW_cluster', 't_PWS_pow_EW_cluster', \
            't_PWS_PWS_EW_cluster', 't_PWS_lin_NS_cluster', \
            't_PWS_pow_NS_cluster', 't_PWS_PWS_NS_cluster', \
            'cc_lin_lin_EW', 'cc_lin_pow_EW', 'cc_lin_PWS_EW', \
            'cc_lin_lin_NS', 'cc_lin_pow_NS', 'cc_lin_PWS_NS', \
            'cc_pow_lin_EW', 'cc_pow_pow_EW', 'cc_pow_PWS_EW', \
            'cc_pow_lin_NS', 'cc_pow_pow_NS', 'cc_pow_PWS_NS', \
            'cc_PWS_lin_EW', 'cc_PWS_pow_EW', 'cc_PWS_PWS_EW', \
            'cc_PWS_lin_NS', 'cc_PWS_pow_NS', 'cc_PWS_PWS_NS', \
            'ratio_lin_lin_EW', 'ratio_lin_pow_EW', \
            'ratio_lin_PWS_EW', 'ratio_lin_lin_NS', \
            'ratio_lin_pow_NS', 'ratio_lin_PWS_NS', \
            'ratio_pow_lin_EW', 'ratio_pow_pow_EW', \
            'ratio_pow_PWS_EW', 'ratio_pow_lin_NS', \
            'ratio_pow_pow_NS', 'ratio_pow_PWS_NS', \
            'ratio_PWS_lin_EW', 'ratio_PWS_pow_EW', \
            'ratio_PWS_PWS_EW', 'ratio_PWS_lin_NS', \
            'ratio_PWS_pow_NS', 'ratio_PWS_PWS_NS', \
            'std_lin_lin_EW', 'std_lin_pow_EW', 'std_lin_PWS_EW', \
            'std_lin_lin_NS', 'std_lin_pow_NS', 'std_lin_PWS_NS', \
            'std_pow_lin_EW', 'std_pow_pow_EW', 'std_pow_PWS_EW', \
            'std_pow_lin_NS', 'std_pow_pow_NS', 'std_pow_PWS_NS', \
            'std_PWS_lin_EW', 'std_PWS_pow_EW', 'std_PWS_PWS_EW', \
            'std_PWS_lin_NS', 'std_PWS_pow_NS', 'std_PWS_PWS_NS', \
            'ntremor_lin_lin', 'ntremor_lin_pow', 'ntremor_lin_PWS', \
            'ntremor_pow_lin', 'ntremor_pow_pow', 'ntremor_pow_PWS', \
            'ntremor_PWS_lin', 'ntremor_PWS_pow', 'ntremor_PWS_PWS'])
                
    # Loop on tremor location
    for i in range(-1, 0): #range(-5, 6):
        for j in range(-5, -2): #range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            filename = '{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}'.format( \
                arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0))

            # Get depth of plate boundary (Preston model)
            myx = depth_pb_P['x'] == x0
            myy = depth_pb_P['y'] == y0
            myline = depth_pb_P[myx & myy]
            d0_P = - myline['depth'].iloc[0]
            # Theoretical time lag
            ts = get_travel_time(sqrt(x0 ** 2 + y0 ** 2), d0_P, h0, vs0)
            tp = get_travel_time(sqrt(x0 ** 2 + y0 ** 2), d0_P, h0, vp0)
            time_P = ts - tp
            # Get computed time lag
            myx = df_width['i'] == i
            myy = df_width['j'] == j
            myline = df_width[myx & myy]
            if myline['maxE'].iloc[0] >= myline['maxN'].iloc[0]:
                timelag = myline['time_EW'].iloc[0]
            else:
                timelag = myline['time_NS'].iloc[0]

            print(x0, y0, time_P, timelag)

            # Get the depth of the plate boundary
            myx = depth['x'] == x0
            myy = depth['y'] == y0
            myline = depth[myx & myy]
            d0 = myline['depth'].iloc[0]

            # Compute the expected timelag between P-wave and S-wave
            distance = sqrt(x0 ** 2.0 + y0 ** 2.0 + d0 ** 2.0)
            tlag = distance * (1.0 / Vs - 1.0 / Vp)
            tbegin = tlag - 1.0
            tend = tlag + 1.0
        
            # Get the number of tremor
            try:
                data = pickle.load(open('cc/' + filename + '_lin.pkl', 'rb'))
                nlincc = len(data[6])
                data = pickle.load(open('cc/' + filename + '_pow.pkl', 'rb'))
                npowcc = len(data[6])
                data = pickle.load(open('cc/' + filename + '_PWS.pkl', 'rb'))
                nPWScc = len(data[6])
#                data = pickle.load(open('ac/' + filename + '_lin.pkl', 'rb'))
#                nlinac = len(data[6])
#                data = pickle.load(open('ac/' + filename + '_pow.pkl', 'rb'))
#                npowac = len(data[6])
#                data = pickle.load(open('ac/' + filename + '_PWS.pkl', 'rb'))
#                nPWSac = len(data[6])

                # If there are tremor at this location
                if (nlincc == npowcc) and (nlincc == nPWScc):
#                    and (nlincc == nlinac) and (nlincc == npowac) \
#                    and (nlincc == nPWSac)):
                    ntremor = nlincc
                    n1 = 0
                    n2 = ntremor
                    EW_UD = data[6]
                    dt = EW_UD[0].stats.delta
                    ncor = int((EW_UD[0].stats.npts - 1) / 2)
                    t = dt * np.arange(- ncor, ncor + 1)
                    ibegin = int(ncor + tbegin / dt)
                    iend = int(ncor + tend / dt)

                    # Plot auto and cross-correlation for linear stack
#                    type_stack = 'lin'
#                    amp = 10.0
#                    amp_lin = 100.0
#                    amp_pow = 2.0
#                    amp_PWS = 200.0
#                    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)
#                    plot_stack_acorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)

                    # Plot auto and cross-correlation for power stack
#                    type_stack = 'pow'
#                    amp = 2.0
#                    amp_lin = 15.0
#                    amp_pow = 1.0
#                    amp_PWS = 100.0
#                    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)
#                    plot_stack_acorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)

                    # Plot auto and cross-correlation for phase-weighted stack
#                    type_stack = 'PWS'
#                    amp = 20.0
#                    amp_lin = 200.0
#                    amp_pow = 10.0
#                    amp_PWS = 1000.0
#                    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)
#                    plot_stack_acorr(arrayName, x0, y0, type_stack, w, \
#                        Tmax, amp, amp_lin, amp_pow, amp_PWS, n1, n2)

                    # Find time of maximum cross-correlation
                    # Linear stack
                    data = pickle.load(open('cc/' + filename + \
                        '_lin.pkl', 'rb'))
                    EW_UD = data[6]
                    NS_UD = data[7]
                    EW_lin = linstack([EW_UD], normalize=False)[0]
                    NS_lin = linstack([NS_UD], normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
                    t_lin_lin_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
                    t_lin_lin_NS = t[ibegin:iend][i0]

                    EW_pow = powstack([EW_UD], w, normalize=False)[0]
                    NS_pow = powstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
                    t_lin_pow_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
                    t_lin_pow_NS = t[ibegin:iend][i0]
            
                    EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
                    NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
                    t_lin_PWS_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
                    t_lin_PWS_NS = t[ibegin:iend][i0]

                    # Power stack
                    data = pickle.load(open('cc/' + filename + \
                        '_pow.pkl', 'rb'))
                    EW_UD = data[6]
                    NS_UD = data[7]
                    EW_lin = linstack([EW_UD], normalize=False)[0]
                    NS_lin = linstack([NS_UD], normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
                    t_pow_lin_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
                    t_pow_lin_NS = t[ibegin:iend][i0]

                    EW_pow = powstack([EW_UD], w, normalize=False)[0]
                    NS_pow = powstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
                    t_pow_pow_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
                    t_pow_pow_NS = t[ibegin:iend][i0]
            
                    EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
                    NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
                    t_pow_PWS_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
                    t_pow_PWS_NS = t[ibegin:iend][i0]

                    # Phase-weighted stack
                    data = pickle.load(open('cc/' + filename + \
                        '_PWS.pkl', 'rb'))
                    EW_UD = data[6]
                    NS_UD = data[7]
                    EW_lin = linstack([EW_UD], normalize=False)[0]
                    NS_lin = linstack([NS_UD], normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
                    t_PWS_lin_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
                    t_PWS_lin_NS = t[ibegin:iend][i0]

                    EW_pow = powstack([EW_UD], w, normalize=False)[0]
                    NS_pow = powstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
                    t_PWS_pow_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
                    t_PWS_pow_NS = t[ibegin:iend][i0]
            
                    EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
                    NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
                    i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
                    t_PWS_PWS_EW = t[ibegin:iend][i0]
                    i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
                    t_PWS_PWS_NS = t[ibegin:iend][i0]
 
                    Tmin = min([t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                                t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                                t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                                t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                                t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                                t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS]) \
                        - 1.0
                    Tmax = max([t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                                t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                                t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                                t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                                t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                                t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS]) \
                        + 1.0

                    # Plot stacks
                    params = {'legend.fontsize': 24, \
                              'xtick.labelsize': 24, \
                              'ytick.labelsize': 24}
                    pylab.rcParams.update(params)
                    plt.figure(0, figsize=(10, 8))
                    plt.axvline(timelag, color='black')
                    plt.axvline(time_P, color='black', linestyle=':')
                    plt.axvline(Tmin, color='grey', linestyle='--')
                    plt.axvline(Tmax, color='grey', linestyle='--')
                    plt.plot(t, EW_PWS.data, 'r-', label='EW')
                    plt.plot(t, NS_PWS.data, 'b-', label='NS')
                    plt.xlim(xmin, xmax)
                    origin = int((len(EW_PWS.data) - 1) / 2)
                    dt = EW_PWS.stats.delta
                    i1 = origin + int(Tmin / EW_PWS.stats.delta)
                    i2 = origin + int(Tmax / EW_PWS.stats.delta) + 1
                    ymin = min(np.min(EW_PWS.data[i1:i2]), np.min(NS_PWS.data[i1:i2]))
                    ymax = max(np.max(EW_PWS.data[i1:i2]), np.max(NS_PWS.data[i1:i2]))
                    plt.ylim(ymin, ymax)
                    plt.title('Stacks for {} at ({:d} - {:d}) km)'.format( \
                        arrayName, int(x0), int(y0)), fontsize=24)
                    plt.xlabel('Lag time (s)', fontsize=24)
                    plt.legend(loc=1)
                    plt.tight_layout()
                    plt.savefig( 'intervals/{}_{:03d}_{:03d}_stacks.eps'. \
                        format(arrayName, int(x0), int(y0)), format='eps')
                    plt.close(0)

                    # Cluster tremor for better peak
                    if (nlincc >= 2):
                    # Linear stack
                        amp = 10.0
                        (clusters, t_lin_lin_EW_cluster, \
                                   t_lin_lin_NS_cluster, \
                            cc_lin_lin_EW, cc_lin_lin_NS, \
                            ratio_lin_lin_EW, ratio_lin_lin_NS,
                            std_lin_lin_EW, std_lin_lin_NS, \
                            ntremor_lin_lin) = cluster_select(arrayName, \
                            x0, y0, 'lin', w, 'lin', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.06, 0.06, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_lin_pow_EW_cluster, \
                                   t_lin_pow_NS_cluster, \
                            cc_lin_pow_EW, cc_lin_pow_NS, \
                            ratio_lin_pow_EW, ratio_lin_pow_NS, \
                            std_lin_pow_EW, std_lin_pow_NS, \
                            ntremor_lin_pow) = cluster_select(arrayName, \
                            x0, y0, 'lin', w, 'pow', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -2.0, 2.0, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_lin_PWS_EW_cluster, \
                                   t_lin_PWS_NS_cluster, \
                            cc_lin_PWS_EW, cc_lin_PWS_NS, \
                            ratio_lin_PWS_EW, ratio_lin_PWS_NS, \
                            std_lin_PWS_EW, std_lin_PWS_NS, \
                            ntremor_lin_PWS) = cluster_select(arrayName, \
                            x0, y0, 'lin', w, 'PWS', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.04, 0.04, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)

                        # Power stack
                        amp = 2.0
                        (clusters, t_pow_lin_EW_cluster, \
                                   t_pow_lin_NS_cluster, \
                            cc_pow_lin_EW, cc_pow_lin_NS, \
                            ratio_pow_lin_EW, ratio_pow_lin_NS, \
                            std_pow_lin_EW, std_pow_lin_NS, \
                            ntremor_pow_lin) = cluster_select(arrayName, \
                            x0, y0, 'pow', w, 'lin', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.3, 0.3, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_pow_pow_EW_cluster, \
                                   t_pow_pow_NS_cluster, \
                            cc_pow_pow_EW, cc_pow_pow_NS, \
                            ratio_pow_pow_EW, ratio_pow_pow_NS, \
                            std_pow_pow_EW, std_pow_pow_NS, \
                            ntremor_pow_pow) = cluster_select(arrayName, \
                            x0, y0, 'pow', w, 'pow', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -10.0, 10.0, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_pow_PWS_EW_cluster, \
                                   t_pow_PWS_NS_cluster, \
                            cc_pow_PWS_EW, cc_pow_PWS_NS, \
                            ratio_pow_PWS_EW, ratio_pow_PWS_NS, \
                            std_pow_PWS_EW, std_pow_PWS_NS, \
                            ntremor_pow_PWS) = cluster_select(arrayName, \
                            x0, y0, 'pow', w, 'PWS', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.16, 0.16, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)

                        # Phase-weighted stack
                        amp = 100.0
                        (clusters, t_PWS_lin_EW_cluster, \
                                   t_PWS_lin_NS_cluster, \
                            cc_PWS_lin_EW, cc_PWS_lin_NS, \
                            ratio_PWS_lin_EW, ratio_PWS_lin_NS, \
                            std_PWS_lin_EW, std_PWS_lin_NS, \
                            ntremor_PWS_lin) = cluster_select(arrayName, \
                            x0, y0, 'PWS', w, 'lin', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.02, 0.02, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_PWS_pow_EW_cluster, \
                                   t_PWS_pow_NS_cluster, \
                            cc_PWS_pow_EW, cc_PWS_pow_NS, \
                            ratio_PWS_pow_EW, ratio_PWS_pow_NS, \
                            std_PWS_pow_EW, std_PWS_pow_NS, \
                            ntremor_PWS_pow) = cluster_select(arrayName, \
                            x0, y0, 'PWS', w, 'pow', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.4, 0.4, 'kmeans', nc, palette, amp, n1, n2, \
                            False, False, True, False, False, False, False)
                        (clusters, t_PWS_PWS_EW_cluster, \
                                   t_PWS_PWS_NS_cluster, \
                            cc_PWS_PWS_EW, cc_PWS_PWS_NS, \
                            ratio_PWS_PWS_EW, ratio_PWS_PWS_NS, \
                            std_PWS_PWS_EW, std_PWS_PWS_NS, \
                            ntremor_PWS_PWS) = cluster_select(arrayName, \
                            x0, y0, 'PWS', w, 'PWS', ncor_cluster, \
                            Tmin, Tmax, RMSmin, RMSmax, xmin, xmax, \
                            -0.004, 0.021, 'kmeans', nc, palette, amp, n1, n2, \
                            True, True, True, True, False, False, False)

                        i0 = len(df.index)
                        df.loc[i0] = [x0, y0, ntremor, \
                            t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                            t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                            t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                            t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                            t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                            t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS, \
                            t_lin_lin_EW_cluster, t_lin_pow_EW_cluster, \
                            t_lin_PWS_EW_cluster, t_lin_lin_NS_cluster, \
                            t_lin_pow_NS_cluster, t_lin_PWS_NS_cluster, \
                            t_pow_lin_EW_cluster, t_pow_pow_EW_cluster, \
                            t_pow_PWS_EW_cluster, t_pow_lin_NS_cluster, \
                            t_pow_pow_NS_cluster, t_pow_PWS_NS_cluster, \
                            t_PWS_lin_EW_cluster, t_PWS_pow_EW_cluster, \
                            t_PWS_PWS_EW_cluster, t_PWS_lin_NS_cluster, \
                            t_PWS_pow_NS_cluster, t_PWS_PWS_NS_cluster, \
                            cc_lin_lin_EW, cc_lin_pow_EW, cc_lin_PWS_EW, \
                            cc_lin_lin_NS, cc_lin_pow_NS, cc_lin_PWS_NS, \
                            cc_pow_lin_EW, cc_pow_pow_EW, cc_pow_PWS_EW, \
                            cc_pow_lin_NS, cc_pow_pow_NS, cc_pow_PWS_NS, \
                            cc_PWS_lin_EW, cc_PWS_pow_EW, cc_PWS_PWS_EW, \
                            cc_PWS_lin_NS, cc_PWS_pow_NS, cc_PWS_PWS_NS, \
                            ratio_lin_lin_EW, ratio_lin_pow_EW, \
                            ratio_lin_PWS_EW, ratio_lin_lin_NS, \
                            ratio_lin_pow_NS, ratio_lin_PWS_NS, \
                            ratio_pow_lin_EW, ratio_pow_pow_EW, \
                            ratio_pow_PWS_EW, ratio_pow_lin_NS, \
                            ratio_pow_pow_NS, ratio_pow_PWS_NS, \
                            ratio_PWS_lin_EW, ratio_PWS_pow_EW, \
                            ratio_PWS_PWS_EW, ratio_PWS_lin_NS, \
                            ratio_PWS_pow_NS, ratio_PWS_PWS_NS, \
                            std_lin_lin_EW, std_lin_pow_EW, std_lin_PWS_EW, \
                            std_lin_lin_NS, std_lin_pow_NS, std_lin_PWS_NS, \
                            std_pow_lin_EW, std_pow_pow_EW, std_pow_PWS_EW, \
                            std_pow_lin_NS, std_pow_pow_NS, std_pow_PWS_NS, \
                            std_PWS_lin_EW, std_PWS_pow_EW, std_PWS_PWS_EW, \
                            std_PWS_lin_NS, std_PWS_pow_NS, std_PWS_PWS_NS, \
                            ntremor_lin_lin, ntremor_lin_pow, ntremor_lin_PWS, \
                            ntremor_pow_lin, ntremor_pow_pow, ntremor_pow_PWS, \
                            ntremor_PWS_lin, ntremor_PWS_pow, ntremor_PWS_PWS]

                    # There is only one tremor
                    else:
                        i0 = len(df.index)
                        df.loc[i0] = [x0, y0, 1, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            np.nan, np.nan, np.nan, \
                            0, 0, 0, 0, 0, 0, 0, 0, 0]

                # All files do not have the same size
                else:
                    print(x0, y0, nlincc, npowcc, nPWScc, nlinac, npowac, nPWSac)

            # If there is no file (= no tremor)
            except:
                i0 = len(df.index)
                df.loc[i0] = [x0, y0, 0, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    np.nan, np.nan, np.nan, \
                    0, 0, 0, 0, 0, 0, 0, 0, 0]

    df['ntremor'] = df['ntremor'].astype('int')
    df['ntremor_lin_lin'] = df['ntremor_lin_lin'].astype('int')
    df['ntremor_lin_pow'] = df['ntremor_lin_pow'].astype('int')
    df['ntremor_lin_PWS'] = df['ntremor_lin_PWS'].astype('int')
    df['ntremor_pow_lin'] = df['ntremor_pow_lin'].astype('int')
    df['ntremor_pow_pow'] = df['ntremor_pow_pow'].astype('int')
    df['ntremor_pow_PWS'] = df['ntremor_pow_PWS'].astype('int')
    df['ntremor_PWS_lin'] = df['ntremor_PWS_lin'].astype('int')
    df['ntremor_PWS_pow'] = df['ntremor_PWS_pow'].astype('int')
    df['ntremor_PWS_PWS'] = df['ntremor_PWS_PWS'].astype('int')
    pickle.dump(df, open(namefile, 'wb'))
