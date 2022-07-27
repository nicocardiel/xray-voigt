import matplotlib.pyplot as plt
import numpy as np
from scipy.special import voigt_profile


def plot_hist_nvoigt(data, vcd, nbins, params, xlabel='x', labelprefix='',
                     figsizex=6, figsizey=4):
    """Plot histogram and overplot Voigt profiles.

    Parameters
    ----------
    data : numpy array
        Initial data to be displayed in the histogram.
    vcd : VoigtComplexDefinition instance
        Instance of VoigtComplexDefinition.
    nbins : int
        Number of bins for the histogram.
    params : instance of lmfit.Parameters
        Parameters defining the set of Voigt profiles.
    xlabel : str
        String for the x-axis label.
    labelprefix : str
        Prefix for the labels in the plot legend.
    figsizex : float
        Figure size in X direction (inches).
    figsizey : float
        Figure size in Y direction (inches).

    """
    ndata = len(data)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsizex, figsizey))
    ax.hist(data, bins=nbins, density=True)
    ax.plot(data, [0] * ndata, '|', color='k', markersize=30, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability density')

    sigma = params['sigma'].value
    mu1 = params['mu1'].value

    sumareas = np.sum(vcd.areas)

    xmin = min(data)
    xmax = max(data)
    xplot = np.linspace(start=xmin, stop=xmax, num=1000)
    yplot = np.zeros_like(xplot)
    for iline in range(vcd.nvoigt):
        gamma = vcd.fwhms_eV[iline] / 2
        mu = mu1 + vcd.energies_eV[iline] - vcd.energies_eV[0]
        w = vcd.areas[iline] / sumareas
        ydum = w * voigt_profile(xplot - mu, sigma, gamma)
        ax.plot(xplot, ydum, label=labelprefix + ' voigt{}'.format(iline + 1))
        yplot = yplot + ydum
    ax.plot(xplot, yplot, 'k-', linewidth=3, label=labelprefix + ' global')
    ax.legend()
    plt.tight_layout()
    plt.show()
