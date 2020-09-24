"""
This module contains a function to associate a cluster to each
time window where there is tremor
"""

import numpy as np
import os
import pandas as pd
import pickle

from math import pi, cos, sin, sqrt

from date import ymdhms2matlab

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

    # Loop on cells
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds

            # Read cluster file
            filename = 'cc/{}/{}_{:03d}_{:03d}/'.format(arrayName, arrayName, \
                int(x0), int(y0)) + '{}_{:03d}_{:03d}_{}_{}_clusters.pkl'. \
                format(arrayName, int(x0), int(y0), type_stack, cc_stack)
            contents = pickle.load(open(filename, 'rb'))
            year = contents[0]
            month = contents[1]
            day = contents[2]
            hour = contents[3]
            minute = contents[4]
            second = contents[5]
            clusters = contents[6]
            i0 = contents[7]
            j0 = contents[8]
            if i0 != j0:
                print(x0, y0, 'Different best clusters for EW and NS')

            # Find tremor in grid cell
            dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
                sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
            dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
                pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
            lonmid = lon0 + x0 / dx
            latmid = lat0 + y0 / dy
        
            index = np.zeros((len(clusters), 4))
            for k in range(0, len(clusters)):
                index[k, 0] = ymdhms2matlab(year[k], month[k], day[k], \
                    hour[k], minute[k], second[k])
                index[k, 1] = latmid
                index[k, 2] = lonmid
                cluster = clusters[k]
                if i0 == j0:
                    if i0 == 0:
                        if cluster == 0:
                            index[k, 3] = 1
                        elif cluster == 1:
                            index[k, 3] = 0
                        else:
                            raise ValueError( 'There should be two clusters')
                    elif i0 == 1:
                        if cluster == 0:
                            index[k, 3] = 0
                        elif cluster == 1:
                            index[k, 3] = 1
                        else:
                            raise ValueError( 'There should be two clusters')
                    else:
                        raise ValueError('Best cluster should be 0 or 1')

            # Save into file
            filename = 'clusters/' + arrayName + '_{:03d}_{:03d}.pkl'.format( \
                int(x0), int(y0))
            pickle.dump(index, open(filename, 'wb'))

    # Gather all data
    first = True
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            filename = 'clusters/' + arrayName + '_{:03d}_{:03d}.pkl'.format( \
                int(x0), int(y0))
            index = pickle.load(open(filename, 'rb'))
            df_temp = pd.DataFrame(index)
            df_temp.columns = ['time', 'latitude', 'longitude', 'keep']
            if first:
                df = df_temp
                first = False
            else:
                df = pd.concat([df, df_temp], ignore_index=True)
    df.sort_values(by=['time'], inplace=True)
    filename = 'clusters/{}_{}_{}_clusters.txt'. \
        format(arrayName, type_stack, cc_stack)
    tfile = open(filename, 'w')
    tfile.write(df.to_string(header=False, index=False))
    tfile.close()

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BH'
    lat0 = 48.0056818181818
    lon0 = -123.084354545455
    ds = 5.0
    type_stack = 'PWS'
    cc_stack = 'PWS'
    write_clusters(arrayName, lat0, lon0, ds, type_stack, cc_stack)
