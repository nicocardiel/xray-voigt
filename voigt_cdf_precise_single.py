"""
Authors: Nicolás Cardiel (UCM) and Maite Ceballos (IFCA)
Copyright 2021-2022
"""
import mpmath as mp
import numpy as np


def voigt_cdf_precise_single(xvalue, sigma, gamma, loc=0):
    """Precise CDF using the mpmath 2F2 hypergeometric function.

    Note that this function can only be evaluated at a scalar 'xvalue'.

    Parameters
    ----------
    xvalue : float
        Location where the CDF will be evaluated.
    sigma : float
        The standard deviation of the Normal distribution part.
    gamma : float
        The half-width at half-maximum of the Lorentzian distribution
        part.
    loc : float
        Center of the Voigt profile.

    Returns
    -------
    result : float
        The CDF evaluated at 'xvalue'.

    """
    z = ((xvalue - loc) + 1j * gamma) / sigma / np.sqrt(2)
    result = 1 / 2 + np.real(mp.erf(z) / 2 + 1j * z * z / mp.pi *
                             mp.hyp2f2(1, 1, 1.5, 2, -z * z))
    return result
