"""
Scripts to compute thickness of tremor zone for all the grid points
"""

import obspy

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, floor, pi, sin, sqrt

from misc import get_travel_time

def MAD(data):
    """
    Compute MAD estimator of scale
    (from Rousseeuw and Croux, 1993)
    """
    m = np.median(data)
    res = 1.4826 * np.median(np.abs(data - m))
    return res

def S(data):
    """
    Compute Sn estimator of scale
    (from Rousseeuw and Croux, 1993)
    """
    N = np.shape(data)[0]
    m = np.zeros(N)
    for i in range(0, N):
        m[i] = np.median(np.abs(data[i] - data))
    res = 1.1926 * np.median(m)
    return res

def Q(data):
    """
    Compute MAD estimator of scale
    (from Rousseeuw and Croux, 1993)
    """
    N = np.shape(data)[0]
    diff = np.zeros(int(N * (N - 1) / 2))
    index = 0
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            diff[index] = abs(data[i] - data[j])
            index = index + 1
    diff = np.sort(diff)
    h = int(floor(N / 2) + 1)
    k = int(h * (h - 1) / 2)
    if (k > 1):
        res = diff[k - 1]
        return res
    else:
        return 0.0

def compute_thickness(arrayName, lon0, lat0, type_stack, cc_stack, mintremor, minratio, ds, amp, h0, vs0, vp0):
    """
    """
    # Get depth of plate boundary around the array
    depth_pb_M = pd.read_csv('../data/depth/McCrory/' + arrayName + '_depth.txt', sep=' ', \
        header=None)
    depth_pb_M.columns = ['x', 'y', 'depth']

    # Get number of tremor and ratio peak / RMS
    df = pickle.load(open(arrayName + '_timelag.pkl', 'rb'))

    # Earth's radius and ellipticity
    a = 6378.136
    e = 0.006694470
    
    # Convert kilometers to latitude, longitude
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

    # Dataframe to store depth and thickness of the tremor zone
    df_thick = pd.DataFrame(columns=['i', 'j', 'latitude', 'longitude', 'distance', 'ntremor', 'ratioE', 'ratioN', \
        'STD_EW', 'STD_NS', 'MAD_EW', 'MAD_NS', 'S_EW', 'S_NS', 'Q_EW', 'Q_NS'])

    # Get velocity model
    f = pickle.load(open('ttgrid_p1.pkl', 'rb'))

    # Loop over output files
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            # Get latitude and longitude
            longitude = lon0 + x0 / dx
            latitude = lat0 + y0 / dy
            # Get depth of plate boundary (McCrory model)
            myx = depth_pb_M['x'] == x0
            myy = depth_pb_M['y'] == y0
            myline = depth_pb_M[myx & myy]
            d0_M = - myline['depth'].iloc[0]
            # Get number of tremor and ratio
            myx = df['x0'] == x0
            myy = df['y0'] == y0
            myline = df[myx & myy]
            ntremor = myline['ntremor_' + type_stack + '_' + cc_stack].iloc[0]
            ratioE = myline['ratio_' + type_stack + '_' + cc_stack + '_EW'].iloc[0]
            ratioN = myline['ratio_' + type_stack + '_' + cc_stack + '_NS'].iloc[0]
            # Look only at best
            if ((ntremor >= mintremor) and \
                ((ratioE >= minratio) or (ratioN >= minratio))):
                # Get file
                filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_stacks.pkl'.format( \
                    arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), type_stack, cc_stack)
                # Read file
                data = pickle.load(open(filename, 'rb'))
                EW = data[0]
                NS = data[1]
                # Corresponding cc times
                npts = int((EW.stats.npts - 1) / 2)
                dt = EW.stats.delta
                t = dt * np.arange(- npts, npts + 1)
                # Theoretical depth (= plate boundary)
                ts = get_travel_time(sqrt(x0 ** 2 + y0 ** 2), d0_M, h0, vs0)
                tp = get_travel_time(sqrt(x0 ** 2 + y0 ** 2), d0_M, h0, vp0)
                time_M = ts - tp
                tbegin = time_M - 2.0
                tend = time_M + 2.0
                ibegin = int(npts + tbegin / dt)
                iend = int(npts + tend / dt)
                # Maxima and corresponding times and depths
                EWmax = np.max(np.abs(EW.data[ibegin:iend]))
                NSmax = np.max(np.abs(NS.data[ibegin:iend]))
                if ((EWmax > 0.05 / amp) or (NSmax > 0.05 / amp)):
                    # Get file
                    filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_timelags.pkl'.format( \
                        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), type_stack, cc_stack)
                    # Read timelag file
                    data = pickle.load(open(filename, 'rb'))
                    timelag_clust_EW = data[0]
                    timelag_clust_NS = data[1]
                    # Initialize
                    STDs_EW = []
                    MADs_EW = []
                    Ss_EW = []
                    Qs_EW = []
                    STDs_NS = []
                    MADs_NS = []
                    Ss_NS = []
                    Qs_NS = []
                    # Loop on clusters
                    for k in range(0, len(timelag_clust_EW)):
                        times_EW = timelag_clust_EW[k].to_numpy()
                        times_NS = timelag_clust_NS[k].to_numpy()
                        depths_EW = np.zeros(np.shape(times_EW)[0])
                        depths_NS = np.zeros(np.shape(times_NS)[0])
                        for l in range(0, np.shape(times_EW)[0]):
                            depths_EW[l] = f(sqrt(x0 ** 2.0 + y0 ** 2.0), times_EW[l])[0]
                        for l in range(0, np.shape(times_NS)[0]):
                            depths_NS[l] = f(sqrt(x0 ** 2.0 + y0 ** 2.0), times_NS[l])[0]
                        STDs_EW.append(np.std(depths_EW))
                        STDs_NS.append(np.std(depths_NS))
                        MADs_EW.append(MAD(depths_EW))
                        MADs_NS.append(MAD(depths_NS))
                        Ss_EW.append(S(depths_EW))
                        Ss_NS.append(S(depths_NS))
                        Qs_EW.append(Q(depths_EW))
                        Qs_NS.append(Q(depths_NS))
                    # Keep minimum value
                    STD_EW = min(STDs_EW)
                    STD_NS = min(STDs_NS)
                    MAD_EW = min(MADs_EW)
                    MAD_NS = min(MADs_NS)
                    S_EW = min(Ss_EW)
                    S_NS = min(Ss_NS)
                    Q_EW = min(Qs_EW)
                    Q_NS = min(Qs_NS)
                    # Write to pandas dataframe
                    i0 = len(df_thick.index)
                    df_thick.loc[i0] = [i, j, latitude, longitude, sqrt(x0 ** 2 + y0 ** 2), ntremor, ratioE, ratioN, \
                        STD_EW, STD_NS, MAD_EW, MAD_NS, S_EW, S_NS, Q_EW, Q_NS]

    # Save dataframe
    df_thick['ntremor'] = df_thick['ntremor'].astype('int')
    namefile = 'cc/{}/{}_{}_{}_thick.pkl'.format(arrayName, arrayName, type_stack, cc_stack)
    pickle.dump(df_thick, open(namefile, 'wb'))

if __name__ == '__main__':

#    arrayName = 'BH'
#    lat0 = 48.0056818181818
#    lon0 = -123.084354545455

#    arrayName = 'BS'
#    lat0 = 47.95728
#    lon0 = -122.92866

#    arrayName = 'CL'
#    lat0 = 48.068735
#    lon0 = -122.969935

#    arrayName = 'DR'
#    lat0 = 48.0059272727273
#    lon0 = -123.313118181818

#    arrayName = 'GC'
#    lat0 = 47.9321857142857
#    lon0 = -123.045528571429

#    arrayName = 'LC'
#    lat0 = 48.0554071428571
#    lon0 = -123.210035714286

#    arrayName = 'PA'
#    lat0 = 48.0549384615385
#    lon0 = -123.464415384615

    arrayName = 'TB'
    lat0 = 47.9730357142857
    lon0 = -123.138492857143

    mintremor = 30
    ds = 5.0
    h0 = np.array([0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 51.0, 81.0])
    vp0 = np.array([5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80, 8.00])
    vs0 = 1.01 * vp0 / np.array([1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77])

#    compute_thickness(arrayName, lon0, lat0, 'lin', 'lin', mintremor, 10.0, ds, 15.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'lin', 'pow', mintremor, 50.0, ds, 0.5, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'lin', 'PWS', mintremor, 100.0, ds, 50.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'lin', mintremor, 10.0, ds, 3.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'pow', mintremor, 50.0, ds, 0.1, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'PWS', mintremor, 100.0, ds, 10.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'PWS', 'lin', mintremor, 10.0, ds, 50.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'PWS', 'pow', mintremor, 50.0, ds, 1.0, h0, vs0, vp0)
    compute_thickness(arrayName, lon0, lat0, 'PWS', 'PWS', mintremor, 5.0, ds, 50.0, h0, vs0, vp0)
