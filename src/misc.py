"""
Miscellaneous functions
"""

import numpy as np

def centroid(t, f):
    """
    Compute the centroid of f(t)

    Input:
        type t = 1D numpy array
        t = time
        type f = 1D numpy array
        f = f(t)
    """
    num = np.sum(t * f)
    denom = np.sum(f)
    return (denom / num)    
