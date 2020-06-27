"""
Miscellaneous functions
"""

import numpy as np
import pickle

from math import asin, cos, pi, sin, sqrt, tan
from scipy import interpolate

def centroid(t, f):
    """
    Compute the centroid of f(t)

    Input:
        type t = 1D numpy array
        t = Time
        type f = 1D numpy array
        f = F(t)
    """
    num = np.sum(t * f)
    denom = np.sum(f)
    return (num / denom)    

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
    
    h0 = np.array([0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 51.0, 81.0])
    vp0 = np.array([5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80, 8.00])
    vs0 = vp0 / np.array([sqrt(3.0), sqrt(3.0), sqrt(3.0), sqrt(3.0), \
        sqrt(3.0), 2.0, sqrt(3.0), sqrt(3.0)])

    f = get_depth_function(h0, vs0, vp0)
    pickle.dump(f, open('ttgrid.pkl', 'wb'))
