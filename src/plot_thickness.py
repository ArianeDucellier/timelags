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

def thickness(dt, x0, y0, time, Vs, Vp):
    """
    Compute corresponding thickness from time delay
    """
    d1 = ((time + dt) / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
    d2 = ((time - dt) / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
    if ((d1 >= 0.0) and (d2>= 0.0)):
        thick = sqrt(d1) - sqrt(d2)
    else:
        thick = np.nan
    return thick

def compute_thickness(arrayName, lon0, lat0, type_stack, cc_stack, mintremor, minratio, Vs, Vp, ds):
    """
    """
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
    df_thick = pd.DataFrame(columns=['latitude', 'longitude', 'distance', 'ntremor', 'ratioE', 'ratioN', \
        'STD_thick_EW', 'STD_thick_NS', 'MAD_thick_EW', 'MAD_thick_NS', \
        'S_thick_EW', 'S_thick_NS', 'Q_thick_EW', 'Q_thick_NS'])

    # Loop over output files
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            # Get latitude and longitude
            longitude = lon0 + x0 / dx
            latitude = lat0 + y0 / dy
            # Get number of tremor and ratio
            myx = df['x0'] == x0
            myy = df['y0'] == y0
            myline = df[myx & myy]
            ntremor = myline['ntremor'].iloc[0]
            ratioE = myline['ratio_' + type_stack + '_' + cc_stack + '_EW'].iloc[0]
            ratioN = myline['ratio_' + type_stack + '_' + cc_stack + '_NS'].iloc[0]
            # Look only at best
            if ((ntremor >= mintremor) and \
                ((ratioE >= minratio) or (ratioN >= minratio))):
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
                    STDs_EW.append(np.std(times_EW))
                    STDs_NS.append(np.std(times_NS))
                    MADs_EW.append(MAD(times_EW))
                    MADs_NS.append(MAD(times_NS))
                    Ss_EW.append(S(times_EW))
                    Ss_NS.append(S(times_NS))
                    Qs_EW.append(Q(times_EW))
                    Qs_NS.append(Q(times_NS))
                # Keep minimum value
                STD_EW = min(STDs_EW)
                STD_NS = min(STDs_NS)
                MAD_EW = min(MADs_EW)
                MAD_NS = min(MADs_NS)
                S_EW = min(Ss_EW)
                S_NS = min(Ss_NS)
                Q_EW = min(Qs_EW)
                Q_NS = min(Qs_NS)
                # Compute corresponding thickness
                if (ratioE > ratioN):
                    time = myline['t_' + type_stack + '_' + cc_stack + '_EW_cluster'].iloc[0]
                else:
                    time = myline['t_' + type_stack + '_' + cc_stack + '_NS_cluster'].iloc[0]
                STD_thick_EW = thickness(STD_EW, x0, y0, time, Vs, Vp)
                STD_thick_NS = thickness(STD_NS, x0, y0, time, Vs, Vp)
                MAD_thick_EW = thickness(MAD_EW, x0, y0, time, Vs, Vp)
                MAD_thick_NS = thickness(MAD_NS, x0, y0, time, Vs, Vp)
                S_thick_EW = thickness(S_EW, x0, y0, time, Vs, Vp)
                S_thick_NS = thickness(S_NS, x0, y0, time, Vs, Vp)
                Q_thick_EW = thickness(Q_EW, x0, y0, time, Vs, Vp)
                Q_thick_NS = thickness(Q_NS, x0, y0, time, Vs, Vp)
                # Write to pandas dataframe
                i0 = len(df_thick.index)
                df_thick.loc[i0] = [latitude, longitude, sqrt(x0 ** 2 + y0 ** 2), ntremor, ratioE, ratioN, \
                    STD_thick_EW, STD_thick_NS, MAD_thick_EW, MAD_thick_NS, \
                    S_thick_EW, S_thick_NS, Q_thick_EW, Q_thick_NS]

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
    Vs = 3.6
    Vp = 6.4
    ds = 5.0

    compute_thickness(arrayName, lon0, lat0, 'lin', 'lin', mintremor, 10.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'lin', 'pow', mintremor, 10.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'lin', 'PWS', mintremor, 50.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'pow', 'lin', mintremor, 10.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'pow', 'pow', mintremor, 30.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'pow', 'PWS', mintremor, 50.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'PWS', 'lin', mintremor, 15.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'PWS', 'pow', mintremor, 40.0, Vs, Vp, ds)
    compute_thickness(arrayName, lon0, lat0, 'PWS', 'PWS', mintremor, 100.0, Vs, Vp, ds)