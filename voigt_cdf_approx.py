import numpy as np
from scipy.interpolate import interp1d
from scipy.special import voigt_profile

from voigt_cdf_precise_single import voigt_cdf_precise_single


def voigt_cdf_approx(xarray, sigma, gamma, loc=None, xmin=None, xmax=None,
                     npoints=10001):
    """Approximate computation of the CDF of a Voigt profile.

    This function can be evaluated over an input array 'xarray'.

    Parameters
    ----------
    xarray : numpy array
        Locations where the CDF will be evaluated.
    sigma : float
        The standard deviation of the Normal distribution part.
    gamma : float
        The half-width at half-maximum of the Lorentzian distribution part.
    loc : float
        Center of the Voigt profile.
    xmin: float or None
        Minimum energy where the Voigt profile will be sampled.
    xmax: float or None
        Maximum energy where the Voigt profile will be sampled.
    npoints : int
        Number of samples for the [xmin, xmax] interval.

    Returns
    -------
    result : scalar or numpy array
        The CDF evaluated at 'xarray'.
    """
    if xmin is None:
        xmin = min(xarray)
    else:
        if np.any(xarray < xmin):
            raise SystemExit('xarray values must be > xmin')
    if xmax is None:
        xmax = max(xarray)
    else:
        if np.any(xarray > xmax):
            raise SystemExit('xarray values must be < xmax')

    xrange = xmax - xmin
    xmin = xmin - xrange / 100
    xmax = xmax + xrange / 100
    xsample = np.linspace(xmin, xmax, npoints)
    deltax = (xmax - xmin) / (npoints - 1)

    # evaluate the Voigt profile at each point of the sampled x interval
    vprofile = voigt_profile(xsample - loc, sigma, gamma)

    # approximate CDF using trapezoidal integration
    ytable = np.cumsum(vprofile * deltax)

    # remove the area corresponding to half of the last bin
    ytable = ytable - 0.5 * vprofile * deltax

    # correct for the real CDF value at the left border of the lower bin
    yoffset = voigt_cdf_precise_single(xmin - 0.5 * deltax,
                                       sigma, gamma, loc=loc)
    ytable = ytable + yoffset

    # generate interpolation function with the computed values
    fun = interp1d(xsample, ytable, kind='linear',
                   assume_sorted=True, bounds_error=True)

    # evaluate CDF at the input array
    return fun(xarray)
