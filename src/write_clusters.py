"""
This module contains a function to associate a cluster to each
time window where there is tremor
"""

import numpy as np
import os
import pickle
import sys
import time

from math import pi, cos, sin, sqrt
from scipy.io import loadmat

from date import matlab2ymdhms

def write_clusters(arrayName, lat0, lon0, ds, type_stack, cc_stack):
    """
    This function associate each one-minute time window where there is a
    tremor to a cluster ('good' or 'bad')
    
    Input:
        type arrayName = string
        arrayName = Name of seismic array
        type lat0 = float
        lat0 = Latitude of the center of the array
        type lon0 = float
        lon0 = Longitude of the center of the array
        type ds = float
        ds = Size of the cell where we look for tremor
        type x0 = float
        x0 = Distance of the center of the cell from the array (east)
        type y0 = float
        y0 = Distance of the center of the cell from the array (north)
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS')
        type cc_stack = string
        cc_stack = Type of stack ('lin', 'pow', 'PWS') over tremor windows
    Output:
        None
    """
    # Earth's radius and ellipticity
    a = 6378.136
    e = 0.006694470

    # Tremor (2009 - 2010)
    data1 = loadmat('../data/mbbp_cat_d_forHeidi')
    mbbp1 = data1['mbbp_cat_d']
    lat1 = mbbp1[:, 2]
    lon1 = mbbp1[:, 3]

    # Tremor (August - September 2011)
    data2 = loadmat('../data/mbbp_ets12')
    mbbp2 = data2['mbbp_ets12']
    lat2 = mbbp2[:, 2]
    lon2 = mbbp2[:, 3]

    # Create indices of clusters
    index1 = np.zeros((np.shape(mbbp1)[0], 4))
    index2 = np.zeros((np.shape(mbbp2)[0], 4))
    index1[:, 3] = np.repeat(-1, np.shape(mbbp1)[0])
    index2[:, 3] = np.repeat(-1, np.shape(mbbp2)[0])

    # Loop on cells
    for i in range(-5, -4):
        for j in range(-5, 1):
            x0 = i * ds
            y0 = j * ds

            # Read cluster file
            filename = 'cc/{}/{}_{:03d}_{:03d}/'.format(arrayName, arrayName, \
                int(x0), int(y0)) + '{}_{:03d}_{:03d}_{}_{}_clusters.pkl'. \
                format(arrayName, int(x0), int(y0), type_stack, cc_stack)
            contents = pickle.load(open(filename, 'rb'))
            i0 = contents[0]
            j0 = contents[1]
            if i0 != j0:
                print(x0, y0, 'Different best clusters for EW and NS')
            clusters = contents[2]

            # Find tremor in grid celle
            dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
                sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
            dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
                pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
            lonmin = lon0 + (x0 - 0.5 * ds) / dx
            lonmax = lon0 + (x0 + 0.5 * ds) / dx
            lonmid = lon0 + x0 / dx
            latmin = lat0 + (y0 - 0.5 * ds) / dy
            latmax = lat0 + (y0 + 0.5 * ds) / dy
            latmid = lat0 + y0 / dy
            find1 = np.where((lat1 >= latmin) & (lat1 <= latmax) & \
                             (lon1 >= lonmin) & (lon1 <= lonmax))
            find2 = np.where((lat2 >= latmin) & (lat2 <= latmax) & \
                             (lon2 >= lonmin) & (lon2 <= lonmax))

            print(len(clusters), len(find1[0]), len(find2[0]))
            # Loop on first data set
            for k in range(0, len(find1[0])):
                index1[find1[0][k], 0] = mbbp1[find1[0][k], 0]
                index1[find1[0][k], 1] = latmid
                index1[find1[0][k], 2] = lonmid
                cluster = clusters[k]
                if i0 == 0:
                    if cluster == 0:
                        index1[find1[0][k], 3] = 1
                    elif cluster == 1:
                        index1[find1[0][k], 3] = 0
                    else:
                        raise ValueError( 'There should be two clusters')
                elif i0 == 1:
                    if cluster == 0:
                        index1[find1[0][k], 3] = 0
                    elif cluster == 1:
                        index1[find1[0][k], 3] = 1
                    else:
                        raise ValueError( 'There should be two clusters')
                else:
                    raise ValueError('Best cluster should be 0 or 1')
                    
            # Loop on second data set
            for k in range(0, len(find2[0])):
                index2[find2[0][k], 0] = mbbp2[find2[0][k], 0]
                index2[find2[0][k], 1] = latmid
                index2[find2[0][k], 2] = lonmid
                cluster = clusters[k + len(find1[0])]
                if i0 == 0:
                    if cluster == 0:
                        index2[find2[0][k], 3] = 1
                    elif cluster == 1:
                        index2[find2[0][k], 3] = 0
                    else:
                        raise ValueError( 'There should be two clusters')
                elif i0 == 1:
                    if cluster == 0:
                        index2[find2[0][k], 3] = 0
                    elif cluster == 1:
                        index2[find2[0][k], 3] = 1
                    else:
                        raise ValueError( 'There should be two clusters')
                else:
                    raise ValueError('Best cluster should be 0 or 1')

    # Save into file
    filename = arrayName + '_clusters_2009-2010.txt'
    np.savetxt(filename, index1)
    filename = arrayName + '_clusters_2011.txt'
    np.savetxt(filename, index2)

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BH'
    lat0 = 48.0056818181818
    lon0 = -123.084354545455
    ds = 5.0
    type_stack = 'PWS'
    cc_stack = 'PWS'
    write_clusters(arrayName, lat0, lon0, ds, type_stack, cc_stack)
