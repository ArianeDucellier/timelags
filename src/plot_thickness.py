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
    # Get relocated tremor locations
    locations = pd.read_csv('../data/Clusters/txt_files/Trem1_Cl2_' + arrayName + '.txt', \
        '\s+', header=None)
    locations.columns = ['longitude', 'latitude', 'lon_reloc', 'lat_reloc']

    # Get depth of plate boundary around the array
    # McCrory model
    depth_pb_M = pd.read_csv('../data/depth/McCrory/' + arrayName + '_depth.txt', sep=' ', \
        header=None)
    depth_pb_M.columns = ['x', 'y', 'depth']
    # Relocated McCrory model
    depth_pb_M_reloc = pd.read_csv('../data/depth/McCrory/' + arrayName + '_depth_reloc.txt', sep=' ', \
        header=None)
    depth_pb_M_reloc.columns = ['longitude', 'latitude', 'lon_reloc', 'lat_reloc', 'depth']

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
    df_thick_reloc = pd.DataFrame(columns=['i', 'j', 'latitude', 'longitude', 'distance', 'ntremor', 'ratioE', 'ratioN', \
        'STD_EW', 'STD_NS', 'MAD_EW', 'MAD_NS', 'S_EW', 'S_NS', 'Q_EW', 'Q_NS'])

    # Get velocity model
    f = pickle.load(open('ttgrid_' + arrayName + '_0.pkl', 'rb'))

    # Loop over output files
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            # Get latitude and longitude
            longitude = lon0 + x0 / dx
            latitude = lat0 + y0 / dy
            # Relocate latitude and longitude
            myx = (locations['longitude'] - longitude > -0.001) & \
                  (locations['longitude'] - longitude < 0.001)
            myy = (locations['latitude'] - latitude > -0.001) & \
                  (locations['latitude'] - latitude < 0.001)
            myline = locations[myx & myy]
            if len(myline > 0):
                lon_reloc = myline['lon_reloc'].iloc[0]
                lat_reloc = myline['lat_reloc'].iloc[0]
                reloc_exist = True
            else:
                reloc_exist = False
            # Get depth of plate boundary (McCrory model)
            myx = depth_pb_M['x'] == x0
            myy = depth_pb_M['y'] == y0
            myline = depth_pb_M[myx & myy]
            d0_M = - myline['depth'].iloc[0]
            # Get relocated depth of plate boundary (McCrory model)
            myx = (depth_pb_M_reloc['longitude'] - longitude > -0.001) & \
                  (depth_pb_M_reloc['longitude'] - longitude < 0.001)
            myy = (depth_pb_M_reloc['latitude'] - latitude > -0.001) & \
                  (depth_pb_M_reloc['latitude'] - latitude < 0.001)
            myline = depth_pb_M_reloc[myx & myy]
            if len(myline > 0):
                d0_M_reloc = - myline['depth'].iloc[0]
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
                    if reloc_exist == True:
                        STDs_EW_reloc = []
                        MADs_EW_reloc = []
                        Ss_EW_reloc = []
                        Qs_EW_reloc = []
                        STDs_NS_reloc = []
                        MADs_NS_reloc = []
                        Ss_NS_reloc = []
                        Qs_NS_reloc = []
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
                        # Depth from A. Wech's catalog
                        if reloc_exist == True:
                            x0_reloc = (lon_reloc - lon0) * dx
                            y0_reloc = (lat_reloc - lat0) * dy
                            depths_EW_reloc = np.zeros(np.shape(times_EW)[0])
                            depths_NS_reloc = np.zeros(np.shape(times_NS)[0])
                            for l in range(0, np.shape(times_EW)[0]):
                                depths_EW_reloc[l] = f(sqrt(x0_reloc ** 2.0 + y0_reloc ** 2.0), times_EW[l])[0]
                            for l in range(0, np.shape(times_NS)[0]):
                                depths_NS_reloc[l] = f(sqrt(x0_reloc ** 2.0 + y0_reloc ** 2.0), times_NS[l])[0]
                        STDs_EW.append(np.std(depths_EW))
                        STDs_NS.append(np.std(depths_NS))
                        MADs_EW.append(MAD(depths_EW))
                        MADs_NS.append(MAD(depths_NS))
                        Ss_EW.append(S(depths_EW))
                        Ss_NS.append(S(depths_NS))
                        Qs_EW.append(Q(depths_EW))
                        Qs_NS.append(Q(depths_NS))
                        if reloc_exist == True:
                            STDs_EW_reloc.append(np.std(depths_EW_reloc))
                            STDs_NS_reloc.append(np.std(depths_NS_reloc))
                            MADs_EW_reloc.append(MAD(depths_EW_reloc))
                            MADs_NS_reloc.append(MAD(depths_NS_reloc))
                            Ss_EW_reloc.append(S(depths_EW_reloc))
                            Ss_NS_reloc.append(S(depths_NS_reloc))
                            Qs_EW_reloc.append(Q(depths_EW_reloc))
                            Qs_NS_reloc.append(Q(depths_NS_reloc))
                    # Keep minimum value
                    STD_EW = min(STDs_EW)
                    STD_NS = min(STDs_NS)
                    MAD_EW = min(MADs_EW)
                    MAD_NS = min(MADs_NS)
                    S_EW = min(Ss_EW)
                    S_NS = min(Ss_NS)
                    Q_EW = min(Qs_EW)
                    Q_NS = min(Qs_NS)
                    if reloc_exist == True:
                        STD_EW_reloc = min(STDs_EW_reloc)
                        STD_NS_reloc = min(STDs_NS_reloc)
                        MAD_EW_reloc = min(MADs_EW_reloc)
                        MAD_NS_reloc = min(MADs_NS_reloc)
                        S_EW_reloc = min(Ss_EW_reloc)
                        S_NS_reloc = min(Ss_NS_reloc)
                        Q_EW_reloc = min(Qs_EW_reloc)
                        Q_NS_reloc = min(Qs_NS_reloc)
                    # Write to pandas dataframe
                    i0 = len(df_thick.index)
                    df_thick.loc[i0] = [i, j, latitude, longitude, sqrt(x0 ** 2 + y0 ** 2), ntremor, ratioE, ratioN, \
                        STD_EW, STD_NS, MAD_EW, MAD_NS, S_EW, S_NS, Q_EW, Q_NS]
                    if reloc_exist == True:
                        i0 = len(df_thick_reloc.index)
                        df_thick_reloc.loc[i0] = [i, j, lat_reloc, lon_reloc, sqrt(x0_reloc ** 2 + y0_reloc ** 2), ntremor, ratioE, ratioN, \
                            STD_EW_reloc, STD_NS_reloc, MAD_EW_reloc, MAD_NS_reloc, S_EW_reloc, S_NS_reloc, Q_EW_reloc, Q_NS_reloc]

    # Save dataframe
    df_thick['ntremor'] = df_thick['ntremor'].astype('int')
    namefile = 'cc/{}/{}_{}_{}_thick.pkl'.format(arrayName, arrayName, type_stack, cc_stack)
    pickle.dump(df_thick, open(namefile, 'wb'))
    df_thick_reloc['ntremor'] = df_thick_reloc['ntremor'].astype('int')
    namefile = 'cc/{}/{}_{}_{}_thick_reloc.pkl'.format(arrayName, arrayName, type_stack, cc_stack)
    pickle.dump(df_thick_reloc, open(namefile, 'wb'))

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

    # 1D velocity model
#    h0 = np.array([0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 51.0, 81.0])
#    vp0 = np.array([5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80, 8.00])
#    vs0 = 1.00 * vp0 / np.array([1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77])

    # Velocity model for BH
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([4.8802, 5.0365, 5.3957, 6.2648, 6.8605, 6.7832, 6.4347, 6.1937, 6.0609, 6.029, 6.1562, 6.5261, 6.8914, 7.2201, 7.5902, 7.99, 7.8729, 7.9578, 8.0181, 8.0238, 8.0381])
#    vs0 = 1.00 * np.array([2.3612, 2.5971, 2.9289, 3.3382, 3.7488, 3.8076, 3.6282, 3.5566, 3.5032, 3.4848, 3.5262, 3.6438, 3.8624, 4.103, 4.316, 4.5194, 4.5373, 4.6046, 4.6349, 4.6381, 4.6461])

    # Velocity model for BS
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.1407, 5.2218, 5.4201, 5.7183, 6.1173, 6.7643, 6.9088, 6.5106, 6.1652, 6.2334, 6.2061, 6.2404, 6.4645, 6.9321, 7.6224, 8.1177, 7.5437, 7.3086, 7.5959, 7.9278, 8.03])
#    vs0 = 1.00 * np.array([2.9093, 3.0335, 3.1533, 3.282, 3.5424, 3.8304, 3.8273, 3.6771, 3.6017, 3.6205, 3.5871, 3.6073, 3.7398, 4.0081, 4.3356, 4.6389, 4.687, 4.4743, 4.4537, 4.5883, 4.6418])

    # Velocity model for CL
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([3.2994, 3.8593, 5.2872, 6.2434, 6.472, 6.5324, 6.8763, 6.9751, 6.6703, 6.74, 6.5739, 6.4228, 6.3528, 6.3597, 6.8624, 7.3501, 7.5318, 7.5597, 7.8917, 7.9976, 8.0119])
#    vs0 = 1.00 * np.array([1.9501, 2.3909, 3.131, 3.5078, 3.6903, 3.7865, 3.8656, 3.9536, 4.0041, 3.883, 3.7767, 3.7125, 3.7202, 3.8272, 4.0085, 4.27, 4.3714, 4.5056, 4.6068, 4.6226, 4.6309])

    # Velocity model for DR
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.0238, 5.8184, 6.493, 6.6639, 6.3367, 6.2555, 6.2872, 6.3607, 6.3724, 6.3109, 6.2716, 6.3945, 6.6777, 7.0275, 7.3712, 7.9404, 8.0719, 8.1048, 8.1162, 8.1019, 8.0919])
#    vs0 = 1.00 * np.array([2.8445, 3.1379, 3.5527, 3.6996, 3.6052, 3.4734, 3.3862, 3.3572, 3.4213, 3.5283, 3.613, 3.7688, 4.082, 4.3885, 4.5483, 4.6166, 4.6516, 4.6851, 4.6917, 4.6831, 4.6771])

    # Velocity model for GC
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.2042, 5.7164, 6.2879, 6.7826, 6.7943, 6.5921, 6.4785, 6.4389, 6.0295, 5.7384, 5.6526, 5.7689, 6.0904, 6.679, 7.3679, 7.7842, 7.563, 7.6962, 8.3223, 8.2024, 8.1181])
#    vs0 = 1.00 * np.array([2.9237, 3.1864, 3.4364, 3.6454, 3.6521, 3.64, 3.6588, 3.565, 3.4499, 3.3791, 3.3742, 3.4807, 3.6942, 4.0203, 4.3638, 4.5984, 4.7217, 4.5042, 4.4379, 4.6202, 4.6927])

    # Velocity model for LC
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([3.5505, 4.4434, 5.9854, 6.6165, 6.8181, 7.1984, 7.0456, 6.8276, 6.6762, 6.5257, 6.4117, 6.4091, 6.5696, 6.867, 7.258, 7.6241, 7.7843, 8.0703, 8.1262, 8.0757, 8.0476])
#    vs0 = 1.00 * np.array([1.9676, 2.5447, 3.4583, 3.833, 3.933, 4.0396, 3.9956, 3.9338, 3.8589, 3.7722, 3.7071, 3.7053, 3.7905, 3.9496, 4.1307, 4.3209, 4.502, 4.6736, 4.6971, 4.668, 4.6516])

    # Velocity model for PA
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([4.2266, 5.1455, 6.1996, 6.3973, 6.5817, 6.6, 6.53, 6.4619, 6.4057, 6.3814, 6.4819, 6.8035, 7.003, 7.1524, 7.4711, 7.8029, 8.0301, 8.1081, 8.1343, 8.1381, 8.1319])
#    vs0 = 1.00 * np.array([2.4303, 2.9904, 3.601, 3.7537, 3.822, 3.815, 3.7748, 3.7351, 3.7024, 3.6885, 3.7567, 4.009, 4.1958, 4.1692, 4.3162, 4.5103, 4.6418, 4.6869, 4.7018, 4.7039, 4.7001])

    # Velocity model for TB
    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
    vp0 = np.array([4.8802, 5.0365, 5.3957, 6.2648, 6.8605, 6.7832, 6.4347, 6.1937, 6.0609, 6.029, 6.1562, 6.5261, 6.8914, 7.2201, 7.5902, 7.99, 7.8729, 7.9578, 8.0181, 8.0238, 8.0381])
    vs0 = 1.00 * np.array([2.3612, 2.5971, 2.9289, 3.3382, 3.7488, 3.8076, 3.6282, 3.5566, 3.5032, 3.4848, 3.5262, 3.6438, 3.8624, 4.103, 4.316, 4.5194, 4.5373, 4.6046, 4.6349, 4.6381, 4.6461])

#    compute_thickness(arrayName, lon0, lat0, 'lin', 'lin', mintremor, 10.0, ds, 15.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'lin', 'pow', mintremor, 50.0, ds, 0.5, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'lin', 'PWS', mintremor, 100.0, ds, 50.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'lin', mintremor, 10.0, ds, 3.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'pow', mintremor, 50.0, ds, 0.1, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'pow', 'PWS', mintremor, 100.0, ds, 10.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'PWS', 'lin', mintremor, 10.0, ds, 50.0, h0, vs0, vp0)
#    compute_thickness(arrayName, lon0, lat0, 'PWS', 'pow', mintremor, 50.0, ds, 1.0, h0, vs0, vp0)
    compute_thickness(arrayName, lon0, lat0, 'PWS', 'PWS', mintremor, 5.0, ds, 50.0, h0, vs0, vp0)
