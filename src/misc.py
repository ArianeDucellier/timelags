"""
Miscellaneous functions
"""

import numpy as np
import pickle

from math import asin, cos, pi, sin, sqrt, tan
from scipy import interpolate

def centroid(t, f, t0, tlag, dt):
    """
    Compute the centroid of f(t)

    Input:
        type t = 1D numpy array
        t = Time
        type f = 1D numpy array
        f = F(t)
    """
    nmax = len(f)
    tbegin = t0 - tlag
    tend = t0 + tlag
    ibegin = max(int(tbegin / dt), 0)
    iend = min(int(tend / dt), nmax)
    num = np.sum(t[ibegin:iend] * f[ibegin:iend])
    denom = np.sum(f[ibegin:iend])
    return (num / denom)

def iterate_centroid(t, f, t0, tlag, dt):
    """
    """
    epsilon = 10.0
    n = 0
    while epsilon > 0.001:
        tnew = centroid(t, f, t0, tlag, dt)
        epsilon = abs(tnew - t0)
        t0 = tnew
        n = n + 1
        if n > 20:
            t0 = np.nan
            break
    return t0
    
def compute_time(h, v, i0):
    """
    Compute the travel time from the
    velocity profile and the incidence angle

    Input:
        type h = 1D numpy array
        h = Thicknesses of the layers
        type v = 1D numpy array
        v = Seismic wave velocities
        type i0 = float
        i0 = Incidence angle
    Output:
        type d0 = float
        d0 = Distance from epicenter
        type t0 = float
        t0 = Travel time
    """
    N = np.shape(h)[0]
    d = np.zeros(N)
    l = np.zeros(N)
    t = np.zeros(N)
    i = np.zeros(N)
    i[0] = i0
    for j in range(0, N):
        d[j] = h[j] * tan(i[j] * pi / 180.0)
        l[j] = h[j] / cos(i[j] * pi / 180.0)
        t[j] = l[j] / v[j]
        if j < N - 1:
            if abs(v[j + 1] * sin(i[j] * pi / 180.0) / v[j]) > 1.0:
                return (10000.0, 10000.0)
            else:
                i[j + 1] = asin(v[j + 1] * sin(i[j] * pi / 180.0) \
                    / v[j]) * 180.0 / pi
    d0 = np.sum(d)
    t0 = np.sum(t)
    return (d0, t0)

def compute_h(depth, v0, h0):
    """
    Compute height and velocity vector from depth

    Input:
        type depth = float
        depth = Depth of seismic source
        type v0 = 1D numpy array
        v0 = Velocity model
        type h0 = 1D numpy array
        h0 = Depth model
    Output:
        type h = 1D numpy array
        h = Thickness input for compute_time
        type v = 1D numpy array
        v = Velocity input for compute_time
    """
    N = 1
    for i in range(1, np.shape(h0)[0]):
        if depth > h0[i]:
            N = N + 1
    h = np.zeros(N)
    v = np.zeros(N)
    for i in range(0, N):
        v[i] = v0[i]
        if (i == N - 1) or (i + 1 >= np.shape(h0)[0]):
            h[i] = depth - h0[i]
        else:
            h[i] = h0[i + 1] - h0[i]
    return (np.flip(h), np.flip(v))
        
def get_travel_time(d0, depth0, h0, v0):
    """
    """
    (h, v) = compute_h(depth0, v0, h0)
    # First round
    i0 = np.arange(0, 90)
    d = np.zeros(90)
    t = np.zeros(90)
    for i in range(0, 90):
        (d[i], t[i]) = compute_time(h, v, i0[i])
    index = np.argmin(np.abs(d0 - d))
    # Second round
    imin = min(0, index - 1)
    imax = max(index + 1, 89)
    i0 = np.arange(imin, imax + 0.1, 0.1)
    d = np.zeros(np.shape(i0)[0])
    t = np.zeros(np.shape(i0)[0])
    for i in range(0, np.shape(i0)[0]):
        (d[i], t[i]) = compute_time(h, v, i0[i])
    index = np.argmin(np.abs(d0 - d))
    t0 = t[index]
    return t0

def get_depth_function(h0, vs0, vp0):
    """
    """
    d = np.arange(0, 45, 5)
    depth = np.arange(30, 60, 2)
    (d, depth) = np.meshgrid(d, depth)
    t = np.zeros((np.shape(d)[0], np.shape(d)[1]))
    for i in range(0, np.shape(t)[0]):
        for j in range(0, np.shape(t)[1]):
            ts = get_travel_time(d[i, j], depth[i, j], h0, vs0)
            tp = get_travel_time(d[i, j], depth[i, j], h0, vp0)
            t[i, j] = ts - tp
    f = interpolate.interp2d(d, t, depth, kind='linear')
    return f

if __name__ == '__main__':
    
    # 1D velocity model
#    h0 = np.array([0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 51.0, 81.0])
#    vp0 = np.array([5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80, 8.00])
#    vs0 = 1.00 * vp0 / np.array([1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77, 1.77])

    # Velocity model for BH
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([4.8802, 5.0365, 5.3957, 6.2648, 6.8605, 6.7832, 6.4347, 6.1937, 6.0609, 6.029, 6.1562, 6.5261, 6.8914, 7.2201, 7.5902, 7.99, 7.8729, 7.9578, 8.0181, 8.0238, 8.0381])
#    vs0 = np.array([2.3612, 2.5971, 2.9289, 3.3382, 3.7488, 3.8076, 3.6282, 3.5566, 3.5032, 3.4848, 3.5262, 3.6438, 3.8624, 4.103, 4.316, 4.5194, 4.5373, 4.6046, 4.6349, 4.6381, 4.6461])

    # Velocity model for BS
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.1407, 5.2218, 5.4201, 5.7183, 6.1173, 6.7643, 6.9088, 6.5106, 6.1652, 6.2334, 6.2061, 6.2404, 6.4645, 6.9321, 7.6224, 8.1177, 7.5437, 7.3086, 7.5959, 7.9278, 8.03])
#    vs0 = np.array([2.9093, 3.0335, 3.1533, 3.282, 3.5424, 3.8304, 3.8273, 3.6771, 3.6017, 3.6205, 3.5871, 3.6073, 3.7398, 4.0081, 4.3356, 4.6389, 4.687, 4.4743, 4.4537, 4.5883, 4.6418])

    # Velocity model for CL
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([3.2994, 3.8593, 5.2872, 6.2434, 6.472, 6.5324, 6.8763, 6.9751, 6.6703, 6.74, 6.5739, 6.4228, 6.3528, 6.3597, 6.8624, 7.3501, 7.5318, 7.5597, 7.8917, 7.9976, 8.0119])
#    vs0 = np.array([1.9501, 2.3909, 3.131, 3.5078, 3.6903, 3.7865, 3.8656, 3.9536, 4.0041, 3.883, 3.7767, 3.7125, 3.7202, 3.8272, 4.0085, 4.27, 4.3714, 4.5056, 4.6068, 4.6226, 4.6309])

    # Velocity model for DR
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.0238, 5.8184, 6.493, 6.6639, 6.3367, 6.2555, 6.2872, 6.3607, 6.3724, 6.3109, 6.2716, 6.3945, 6.6777, 7.0275, 7.3712, 7.9404, 8.0719, 8.1048, 8.1162, 8.1019, 8.0919])
#    vs0 = np.array([2.8445, 3.1379, 3.5527, 3.6996, 3.6052, 3.4734, 3.3862, 3.3572, 3.4213, 3.5283, 3.613, 3.7688, 4.082, 4.3885, 4.5483, 4.6166, 4.6516, 4.6851, 4.6917, 4.6831, 4.6771])

    # Velocity model for GC
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([5.2042, 5.7164, 6.2879, 6.7826, 6.7943, 6.5921, 6.4785, 6.4389, 6.0295, 5.7384, 5.6526, 5.7689, 6.0904, 6.679, 7.3679, 7.7842, 7.563, 7.6962, 8.3223, 8.2024, 8.1181])
#    vs0 = np.array([2.9237, 3.1864, 3.4364, 3.6454, 3.6521, 3.64, 3.6588, 3.565, 3.4499, 3.3791, 3.3742, 3.4807, 3.6942, 4.0203, 4.3638, 4.5984, 4.7217, 4.5042, 4.4379, 4.6202, 4.6927])

    # Velocity model for LC
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([3.5505, 4.4434, 5.9854, 6.6165, 6.8181, 7.1984, 7.0456, 6.8276, 6.6762, 6.5257, 6.4117, 6.4091, 6.5696, 6.867, 7.258, 7.6241, 7.7843, 8.0703, 8.1262, 8.0757, 8.0476])
#    vs0 = np.array([1.9676, 2.5447, 3.4583, 3.833, 3.933, 4.0396, 3.9956, 3.9338, 3.8589, 3.7722, 3.7071, 3.7053, 3.7905, 3.9496, 4.1307, 4.3209, 4.502, 4.6736, 4.6971, 4.668, 4.6516])

    # Velocity model for PA
#    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
#    vp0 = np.array([4.2266, 5.1455, 6.1996, 6.3973, 6.5817, 6.6, 6.53, 6.4619, 6.4057, 6.3814, 6.4819, 6.8035, 7.003, 7.1524, 7.4711, 7.8029, 8.0301, 8.1081, 8.1343, 8.1381, 8.1319])
#    vs0 = np.array([2.4303, 2.9904, 3.601, 3.7537, 3.822, 3.815, 3.7748, 3.7351, 3.7024, 3.6885, 3.7567, 4.009, 4.1958, 4.1692, 4.3162, 4.5103, 4.6418, 4.6869, 4.7018, 4.7039, 4.7001])

    # Velocity model for TB
    h0 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
    vp0 = np.array([4.8802, 5.0365, 5.3957, 6.2648, 6.8605, 6.7832, 6.4347, 6.1937, 6.0609, 6.029, 6.1562, 6.5261, 6.8914, 7.2201, 7.5902, 7.99, 7.8729, 7.9578, 8.0181, 8.0238, 8.0381])
    vs0 = np.array([2.3612, 2.5971, 2.9289, 3.3382, 3.7488, 3.8076, 3.6282, 3.5566, 3.5032, 3.4848, 3.5262, 3.6438, 3.8624, 4.103, 4.316, 4.5194, 4.5373, 4.6046, 4.6349, 4.6381, 4.6461])

    f = get_depth_function(h0, vs0, vp0)
    pickle.dump(f, open('ttgrid_TB.pkl', 'wb'))
