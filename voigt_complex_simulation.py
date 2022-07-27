import datetime
from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from ecdf_nvoigt import ecdf_nvoigt
from plot_hist_nvoigt import plot_hist_nvoigt
from residual_nvoigt import residual_nvoigt
from voigt_cdf_approx import voigt_cdf_approx
from voigt_complex_definition import VoigtComplexDefinition

# Global variables
factor_sigma_fwhm = 2 * np.sqrt(2 * np.log(2))


class VoigtComplexSimulation:
    """Simulation of Voigt Complex.

    Parameters
    ----------
    name : str
        Name of the complex of lines.
    nphotons : int
        Number of photons to be simulated.
    fwhm_g : float
        The Full Width at Half Maximum of the Normal distribution part.
    xmin : float
        Minimum energy.
    xmax : float
        Maximum energy.
    nsampling : int
        Number of points to sample the [xmin, xmax] interval in
        order to generate the intermediate CDF.
    rng : instance of numpy Generator
        Numpy Generator to generate random samples.

    Attributes
    ----------
    vcd: VoigtComplexDefinition instance
        Instance of VoigtComplexDefinition.
    nphotons_simul : int
        Number of simulated photons within the interval [xmin, xmax].
    photons_simul : numpy array
        Energy of simulated photons.
    nleft : int
        Number of photons with energy < xmin. These points do not
        appear in the simulated 'photons_simul' array.
    nright : int
        Number of photons with energy > xmax. These points do not
        appear in the simulated 'photons_simul' array.
    """

    def __init__(self, name, nphotons, fwhm_g, xmin, xmax, nsampling=10001, rng=None):
        vcd = VoigtComplexDefinition(name)
        self.vcd = vcd
        self.nphotons = nphotons
        self.fwhm_g = fwhm_g
        self.xmin = xmin
        self.xmax = xmax
        self.nsampling = nsampling
        if rng is None:
            raise ValueError('Undefined rng parameter')

        sigma = fwhm_g / factor_sigma_fwhm
        xrange = xmax - xmin

        # generate CDF
        sumareas = np.sum(vcd.areas)
        xarray = np.linspace(xmin, xmax, nsampling)
        ycdf_model = np.zeros_like(xarray)
        for iline in range(vcd.nvoigt):
            gamma = vcd.fwhms_eV[iline] / 2  # HWHM
            mu = vcd.energies_eV[iline]
            w = vcd.areas[iline] / sumareas
            ydum = w * voigt_cdf_approx(xarray, sigma=sigma, gamma=gamma, loc=mu)
            ycdf_model = ycdf_model + ydum

        # define interpolation function to invert the CDF
        fun = interp1d(ycdf_model, xarray, kind='linear', assume_sorted=True,
                       bounds_error=False,
                       fill_value=(xmin - xrange, xmax + xrange))

        # random samples from a uniform distribution
        uniform_samples = rng.uniform(low=0, high=1, size=nphotons)

        # generate initial simulated data set
        data = fun(uniform_samples)

        # count and remove data below xmin
        ibelow = np.argwhere(data < xmin)
        nleft = len(ibelow)
        data = np.delete(data, ibelow)

        # count and remove data above xmax
        iabove = np.argwhere(data > xmax)
        nright = len(iabove)
        data = np.delete(data, iabove)

        # store results
        self.nphotons_simul = len(data)
        self.photons_simul = data
        self.nleft = nleft
        self.nright = nright

    def fit_nvoigt_complex(self, fwhm_g_ini=3.0, mu1_ini=5900,
                           nmax_iterations=20,
                           plots=True, nbins=200,
                           figsizex=6, figsizey=4,
                           savepdf=False,
                           verbose=True):
        """Fit MnKa complex using Voigt profiles.

        Parameters
        ----------
        fwhm_g_ini : float
            Initial guess for FWHM_G (eV).
        mu1_ini : float
            Central wavelength guess for first line in the complex (eV).
        nmax_iterations : int
            Maximum number of iterations to refine the nleft and nright
            parameters when estimating the empirical CDF.
        plots : bool
            If True, display intermediate plots.
        nbins : int
            Number of bins to display histograms. This number has no effect
            on the resulting fit.
        figsizex : float
            Figure size in X direction (inches).
        figsizey : float
            Figure size in Y direction (inches).
        savepdf : bool
            If True, save plots in PDF format.
        verbose : bool
            If True, print intermediate information.

        Returns
        -------
        mu1 : float
            Central wavelength of first Mn Ka line
        mu1_err : float
            Uncertainty in 'mu1' provided by lmfit.
        fwhm_g : float
            The Full Width at Half Maximum of the Normal distribution part.
        fwhm_g_err : float
            Uncertainty in 'fwhm_g' provided by lmfit.
        nleft : int
            Number of photons with energy < xmin. These points do not
            appear in the simulated 'photons_simul' array.
        nright : int
            Number of photons with energy > xmax. These points do not
            appear in the simulated 'photons_simul' array.
        """

        # display the initial set of simulated photons
        if plots:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsizex, figsizey))
            ax.hist(self.photons_simul, bins=nbins)
            ax.plot(self.photons_simul, [0] * self.nphotons_simul, '|', color='k', markersize=30, alpha=0.2)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Number of photons')
            ax.set_title(self.vcd.name)
            plt.tight_layout()
            if savepdf:
                plt.savefig(f'test/hist_{self.vcd.name}.pdf')
            plt.show()

        # data rescaling
        datanor = self.photons_simul - self.vcd.global_mean
        ndata = len(datanor)

        # display rescaled data
        if plots:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsizex, figsizey))
            ax.hist(datanor, bins=nbins)
            ax.plot(datanor, [0] * ndata, '|', color='k', markersize=30, alpha=0.2)
            ax.set_xlabel('Rescaled energy')
            ax.set_ylabel('Number of photons')
            ax.set_title(self.vcd.name)
            plt.tight_layout()
            if savepdf:
                plt.savefig(f'test/hist_{self.vcd.name}_rescaled.pdf')
            plt.show()

        # initial solution guess
        mu1_ini_nor = mu1_ini - self.vcd.global_mean
        sigma_ini_nor = fwhm_g_ini / factor_sigma_fwhm
        params = Parameters()
        params.add('sigma', value=sigma_ini_nor, min=0.1, max=10)
        params.add('mu1', value=mu1_ini_nor, min=-10, max=10)
        params.add('nleft', value=0, vary=False, min=0)
        params.add('nright', value=0, vary=False, min=0)

        if plots:
            if savepdf:
                pdfoutput = f'test/hist_{self.vcd.name}_initial_guess.pdf'
            else:
                pdfoutput = None
            plot_hist_nvoigt(data=datanor, vcd=self.vcd, nbins=nbins,
                             params=params,
                             xlabel='Rescaled energy', labelprefix='initial',
                             figsizex=figsizex, figsizey=figsizey,
                             pdfoutput=pdfoutput)

        pdfoutput = None
        if plots:
            if savepdf:
                pdfoutput = f'test/cdf_{self.vcd.name}_initial_guess.pdf'
        xecdf_data, yecdf_data, ycdf_model = ecdf_nvoigt(
            data=datanor,
            vcd=self.vcd,
            params=params,
            modellabel='initial model',
            plot=plots,
            xlabel='Rescaled energy',
            figsizex=figsizex,
            figsizey=figsizey,
            pdfoutput=pdfoutput
        )

        # fitting computation
        iteration = 0
        out = None
        while iteration < nmax_iterations:
            if verbose:
                print('\n* Iteration #{}'.format(iteration))
                params.pretty_print()
            # method='leastsq': Levenberg-Marquard
            time_ini = datetime.datetime.now()
            if verbose:
                print('Minimization starts at: {}'.format(time_ini))
            out = minimize(
                residual_nvoigt,
                params,
                args=(datanor, self.vcd),
                method='leastsq'
            )
            time_end = datetime.datetime.now()
            if verbose:
                print('\n\n* Resulting fitting parameters:')
                print('Minimization ends at..: {}'.format(time_end))
                print('Elapsed time..........: {}'.format(time_end - time_ini))
                out.params.pretty_print()
                print('')
                print(fit_report(out, min_correl=0.0))

            if plots:
                plot_hist_nvoigt(data=datanor, vcd=self.vcd, nbins=nbins,
                                 params=out.params,
                                 xlabel='rescaled energy', labelprefix='fitted',
                                 figsizex=figsizex, figsizey=figsizey)

            xecdf_data, yecdf_data, ycdf_model = ecdf_nvoigt(
                data=datanor,
                vcd=self.vcd,
                params=out.params,
                modellabel='fitted model',
                plot=plots,
                xlabel='Rescaled energy',
                figsizex=figsizex,
                figsizey=figsizey
            )

            if verbose:
                print('yecdf_data: {}'.format(yecdf_data))
                print('ycdf_model: {}'.format(ycdf_model))

            nleft = int((ycdf_model[0] - yecdf_data[0]) * ndata + 0.5)
            nright = int((yecdf_data[-1] - ycdf_model[-1]) * ndata + 0.5)
            if verbose:
                print('nleft correction...: {}'.format(nleft))
                print('nright correction..: {}'.format(nright))
            if nleft == 0 and nright == 0:
                break
            else:
                for item in ['mu1', 'sigma']:
                    params[item].value = out.params[item].value
                params['nleft'].value = out.params['nleft'].value + nleft
                params['nright'].value = out.params['nright'].value + nright
            iteration += 1

        # plot final fit
        mu1 = out.params['mu1'].value + self.vcd.global_mean
        mu1_err = out.params['mu1'].stderr
        sigma = out.params['sigma'].value
        sigma_err = out.params['sigma'].stderr
        nleft = out.params['nleft'].value
        nright = out.params['nright'].value

        newparams = Parameters()
        newparams.add('mu1', value=mu1)
        newparams.add('sigma', value=sigma)
        newparams.add('nleft', value=nleft)
        newparams.add('nright', value=nright)
        newparams['mu1'].stderr = mu1_err
        newparams['sigma'].stderr = sigma_err
        if verbose:
            newparams.pretty_print(colwidth=9, precision=8,
                                   columns=['value', 'stderr'])

        if plots:
            if savepdf:
                pdfoutput = f'test/hist_{self.vcd.name}_final_fit.pdf'
            else:
                pdfoutput = None
            plot_hist_nvoigt(data=self.photons_simul, vcd=self.vcd, nbins=nbins,
                             params=newparams,
                             xlabel='Energy (eV)', labelprefix='fitted',
                             figsizex=figsizex, figsizey=figsizey,
                             pdfoutput=pdfoutput)

        pdfoutput = None
        if plots:
            if savepdf:
                pdfoutput = f'test/cdf_{self.vcd.name}_final_fit.pdf'
        xecdf_data, yecdf_data, ycdf_model = ecdf_nvoigt(
            data=self.photons_simul,
            vcd=self.vcd,
            params=newparams,
            modellabel='fitted model',
            plot=plots,
            xlabel='Energy (eV)',
            figsizex=figsizex,
            figsizey=figsizey,
            pdfoutput=pdfoutput
        )

        fwhm_g = sigma * factor_sigma_fwhm
        fwhm_g_err = sigma_err * factor_sigma_fwhm
        if verbose:
            print('\nFWHM_G: {:.4f} +/- {:.4f} eV\n'.format(fwhm_g, fwhm_g_err))

        return mu1, mu1_err, fwhm_g, fwhm_g_err, nleft, nright
