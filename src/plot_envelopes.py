"""
Scripts to plot envelopes of stacks for all the grid points
"""

import obspy

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sqrt, sin

from misc import centroid

def plot_envelopes(arrayName, lon0, lat0, type_stack, cc_stack, mintremor, \
    minratio, Tmax, amp, Vs, Vp, ds, imin, imax, jmin, jmax, cut):
    """
    """
    # Get depth of plate boundary around the array
    depth_pb = pd.read_csv('../data/depth/' + arrayName + '_depth.txt', sep=' ', \
        header=None)
    depth_pb.columns = ['x', 'y', 'depth']

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

    # Create figure
    params = {'xtick.labelsize':40,
              'ytick.labelsize':40}
    pylab.rcParams.update(params)
    plt.figure(1, figsize=(3 * (imax - imin + 1), 2 * (jmax - jmin + 1)))

    # Dataframe to store difference in time lags between EW and NS
    df_dt = pd.DataFrame(columns=['latitude', 'longitude', 'distance', 'diff_time', 'diff_depth'])

    # Dataframe to store depth and thickness of the tremor zone
    df_width = pd.DataFrame(columns=['latitude', 'longitude', 'distance', 'ntremor', 'ratioE', 'ratioN', \
        'time_EW', 'time_NS', 'dist_EW', 'dist_NS', 'd_to_pb_EW', 'd_to_pb_NS', 'thick_EW', 'thick_NS'])

    # Loop over output files
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            x0 = i * ds
            y0 = j * ds
            # Get latitude and longitude
            longitude = lon0 + x0 / dx
            latitude = lat0 + y0 / dy
            # Get depth
            myx = depth_pb['x'] == x0
            myy = depth_pb['y'] == y0
            myline = depth_pb[myx & myy]
            d0 = myline['depth'].iloc[0]
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
                icut1 = npts + int(cut / dt)
                icut2 = npts + int((Tmax - cut) / dt)
                # Theoretical depth
                dist = sqrt(d0 ** 2.0 + x0 ** 2 + y0 ** 2)
                time = dist * (1.0 / Vs - 1.0 / Vp)
                tbegin = time - 2.0
                tend = time + 2.0
                ibegin = int(npts + tbegin / dt)
                iend = int(npts + tend / dt)
                # Maxima and corresponding times and depths
                EWmax = np.max(np.abs(EW.data[ibegin:iend]))
                NSmax = np.max(np.abs(NS.data[ibegin:iend]))
#                i_EW = np.argmax(np.abs(EW.data[ibegin:iend]))
#                i_NS = np.argmax(np.abs(NS.data[ibegin:iend]))
#                time_EW = t[ibegin:iend][i_EW]
#                time_NS = t[ibegin:iend][i_NS]
                time_EW = centroid(t[ibegin:iend], EW.data[ibegin:iend])
                time_NS = centroid(t[ibegin:iend], NS.data[ibegin:iend])
                distance = (time_EW / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dist_EW = sqrt(distance)
                    d_to_pb_EW = d0 + dist_EW
                else:
                    dist_EW = np.nan
                    d_to_pb_EW = np.nan
                distance = (time_NS / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dist_NS = sqrt(distance)
                    d_to_pb_NS = d0 + dist_NS
                else:
                    dist_NS = np.nan
                    d_to_pb_NS = np.nan
                # Time difference and depth difference
                diff_time = time_EW - time_NS
                diff_depth = dist_EW - dist_NS
                # Write to pandas dataframe
                if ((ratioE >= minratio) and (ratioN >= minratio)):
                    i0 = len(df_dt.index)
                    df_dt.loc[i0] = [latitude, longitude, sqrt(x0 ** 2 + y0 ** 2), diff_time, diff_depth]
                # Compute width of envelope at half the amplitude of the maximum
                indEW = np.where(EW.data[ibegin:iend] > 0.5 * EWmax)[0]
                indNS = np.where(NS.data[ibegin:iend] > 0.5 * NSmax)[0]
                timerange_EW = t[ibegin:iend][indEW]
                timerange_NS = t[ibegin:iend][indNS]
                tmax_EW = np.max(timerange_EW)
                tmin_EW = np.min(timerange_EW)
                tmax_NS = np.max(timerange_NS)
                tmin_NS = np.min(timerange_NS)
                distance = (tmax_EW / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dmax_EW = sqrt(distance)
                else:
                    dmax_EW = np.nan
                distance = (tmin_EW / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dmin_EW = sqrt(distance)
                else:
                    dmin_EW = np.nan
                distance = (tmax_NS / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dmax_NS = sqrt(distance)
                else:
                    dmax_NS = np.nan
                distance = (tmin_NS / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - x0 ** 2.0 - y0 ** 2.0
                if (distance >= 0.0):
                    dmin_NS = sqrt(distance)
                else:
                    dmin_NS = np.nan
                thick_EW = dmax_EW - dmin_EW
                thick_NS = dmax_NS - dmin_NS
                # Write to pandas dataframe
                i0 = len(df_width.index)
                df_width.loc[i0] = [latitude, longitude, sqrt(x0 ** 2 + y0 ** 2), ntremor, ratioE, ratioN, \
                    time_EW, time_NS, dist_EW, dist_NS, d_to_pb_EW, d_to_pb_NS, thick_EW, thick_NS]
                # Plot
                plt.plot([(i - imin) * Tmax + time, (i - imin) * Tmax + time], \
                         [(j - jmin), (j - jmin + 0.8)], linewidth=3, color='grey')
                plt.plot((i - imin) * Tmax + t[icut1 : icut2], \
                    (j - jmin) + amp * EW.data[icut1 : icut2], 'r-')
                plt.plot((i - imin) * Tmax + t[icut1 : icut2], \
                    (j - jmin) + amp * NS.data[icut1 : icut2], 'b-')
                plt.annotate('{}'.format(ntremor), \
                    ((i - imin + 0.6) * Tmax, (j - jmin + 0.5)), fontsize=30)

    # Finalize figure
    plt.xlim(0, Tmax * (imax - imin + 1))
    plt.ylim(-0.2, jmax - jmin + 1)
    xlabels = []
    for i in range(imin, imax + 1):
        xlabels.append('{:.0f}'.format(i * ds))
    ylabels = []
    for j in range(jmin, jmax + 1):
        ylabels.append('{:.0f}'.format(j * ds))
    plt.xticks(Tmax * (0.5 + np.arange(0, imax - imin + 1)), xlabels)
    plt.yticks(0.5 + np.arange(0, jmax - jmin + 1), ylabels)
    plt.xlabel('Distance from the array (km) in east direction', fontsize=40)
    plt.ylabel('Distance from the array (km) in north direction', fontsize=40)
    plt.savefig('cc/{}/{}_{}_{}.eps'.format(arrayName, arrayName, type_stack, cc_stack), format='eps')
    plt.close(1)

    # Save dataframe
    namefile = 'cc/{}/{}_{}_{}_diff_EW_NS.pkl'.format(arrayName, arrayName, type_stack, cc_stack)
    pickle.dump(df_dt, open(namefile, 'wb'))
    df_width['ntremor'] = df_width['ntremor'].astype('int')
    namefile = 'cc/{}/{}_{}_{}_width.pkl'.format(arrayName, arrayName, type_stack, cc_stack)
    pickle.dump(df_width, open(namefile, 'wb'))

if __name__ == '__main__':

#    arrayName = 'BH'
#    lat0 = 48.0056818181818
#    lon0 = -123.084354545455

    arrayName = 'BS'
    lat0 = 47.95728
    lon0 = -122.92866

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

#    arrayName = 'TB'
#    lat0 = 47.9730357142857
#    lon0 = -123.138492857143

    mintremor = 30
    Tmax = 15.0
    Vs = 3.6
    Vp = 6.4
    ds = 5.0
    imin = -5
    imax = 0
    jmin = -4
    jmax = 2
    cut = 2.0

#    plot_envelopes(arrayName, lon0, lat0, 'lin', 'lin', mintremor, 10.0, Tmax, 0.1, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'lin', 'pow', mintremor, 10.0, Tmax, 3.0, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'lin', 'PWS', mintremor, 50.0, Tmax, 0.05, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'pow', 'lin', mintremor, 10.0, Tmax, 0.5, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'pow', 'pow', mintremor, 30.0, Tmax, 10.0, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'pow', 'PWS', mintremor, 50.0, Tmax, 0.2, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'PWS', 'lin', mintremor, 15.0, Tmax, 0.05, Vs, Vp, ds, imin, imax, jmin, jmax)
#    plot_envelopes(arrayName, lon0, lat0, 'PWS', 'pow', mintremor, 40.0, Tmax, 1.0, Vs, Vp, ds, imin, imax, jmin, jmax)
    plot_envelopes(arrayName, lon0, lat0, 'PWS', 'PWS', mintremor, 150.0, \
        Tmax, 100.0, Vs, Vp, ds, imin, imax, jmin, jmax, cut)
