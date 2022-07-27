import matplotlib.pyplot as plt
import numpy as np

from ecdf import ecdf
from voigt_cdf_approx import voigt_cdf_approx


def ecdf_nvoigt(data, vcd, params, modellabel='model', plot=True, alpha=1.0,
                xlabel='Energy', figsizex=6, figsizey=4, pdfoutput=None):
    """Compute empirical and model CDF.

    Parameters
    ----------
    data : numpy array
        Initial data to be displayed in the histogram.
    vcd : VoigtComplexDefinition instance
        Instance of VoigtComplexDefinition.
    params : instance of lmfit.Parameters
        Parameters defining the set of Voigt profiles.
    modellabel : str
        Label for model CDF.
    plot : bool
        If True, plot empirical and model CDF.
    alpha : float
        Transparency for symbols in plot.
    xlabel : str
        Label for X axis.
    figsizex : float
        Figure size in X direction (inches).
    figsizey : float
        Figure size in Y direction (inches).
    pdfoutput : str or None
        Name of the PDF file to store the plot.

    Returns
    -------
    xecdf_data : numpy array
        Sorted input 'data' array.
    yecdf_data : numpy array
        ECDF values associated to the sorted 'data' array.
    ycdf_model : numpy array
        CDF associated to the model.
    """

    nleft = int(params['nleft'].value + 0.5)
    nright = int(params['nright'].value + 0.5)

    # compute ECDF of data
    xecdf_data, yecdf_data = ecdf(data=data, nleft=nleft, nright=nright)

    # compute CDF of model
    sigma = params['sigma'].value
    mu1 = params['mu1'].value

    sumareas = np.sum(vcd.areas)

    ycdf_model = np.zeros_like(xecdf_data)
    for iline in range(vcd.nvoigt):
        gamma = vcd.fwhms_eV[iline] / 2
        mu = mu1 + vcd.energies_eV[iline] - vcd.energies_eV[0]
        w = vcd.areas[iline] / sumareas
        ydum = w * voigt_cdf_approx(xecdf_data, sigma=sigma, gamma=gamma, loc=mu)
        ycdf_model = ycdf_model + ydum

    if plot:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsizex, figsizey))
        ax.plot(xecdf_data, yecdf_data, 'C1.', alpha=alpha, label='original data')
        ax.plot(xecdf_data, ycdf_model, 'k-', alpha=alpha, label=modellabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        plt.tight_layout()
        if pdfoutput is not None:
            plt.savefig(pdfoutput)
        plt.show()

    return xecdf_data, yecdf_data, ycdf_model
