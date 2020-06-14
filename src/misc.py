"""
Miscellaneous functions
"""

import numpy as np

from math import asin, cos, pi, sin, tan

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
    return (denom / num)    

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
            i[j + 1] = asin(v[j + 1] * sin(i[j] * pi / 180.0) \
                / v[j]) * 180.0 / pi
    d0 = np.sum(d)
    t0 = np.sum(t)
    return (d, t)

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
    return (h, v)
        
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
  
#vp_vs=sqrt(3);
#vel_P2=[ 5.40  0.0; 6.38  4.0; 6.59  9.0;   6.73 16.0;   6.86 20.0;   6.95 25.0;   7.80 51.0;   8.00 81.0];