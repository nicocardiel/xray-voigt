import numpy as np


def ecdf(data, nleft=0, nright=0):
    """Empirical Cumulative Distribution Function.

    Parameters
    ----------
    data : numpy array
        Array of empirical values.
    nleft : int
        Number of data points to the left of the range exhibited by
        the 'data' array.
    nright : int
        Number of data points to the right for the range exhibited by
        the 'data' array.

    Returns
    -------
    x : numpy array
        Sorted input 'data' array
    y : numpy array
        ECDF values associated to the sorted 'data' array.
    """
    x = np.sort(data)
    n = x.size
    ntot = n + nleft + nright
    y = np.arange(1 + nleft, n + 1 + nleft) / ntot
    return x, y
