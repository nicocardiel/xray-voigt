"""Simulate Voigt complexes.

Authors: Nicolás Cardiel and Maite Ceballos
"""

import argparse
from astropy.io import fits
import numpy as np
from numpy.random import default_rng

from voigt_complex_simulation import VoigtComplexSimulation


def main():
    # parse command-line options
    parser = argparse.ArgumentParser(description="Simulate Voigt complexes")
    parser.add_argument("--name", help="complex name", type=str, default="MnKa")
    parser.add_argument("--nphotini_min", help="minimum initial number of photons per simulation",
                        type=int, default=4000)
    parser.add_argument("--nphotini_max", help="maximum initial number of photons per simulation",
                        type=int, default=16000)
    parser.add_argument("--nphotini_nstep", help=" number of steps for initial number of photons",
                        type=int, default=7)
    parser.add_argument("--nsimulations", help="total number of simulations/nphotini",
                        type=int, default=1000)
    parser.add_argument("--fwhm_g", help="FWHM of the Gaussian part", type=float, default=2.2)
    parser.add_argument("--xmin", help="minimum energy", type=float, default=5860)
    parser.add_argument("--xmax", help="maximum energy", type=float, default=5920)
    parser.add_argument("--seed", help="seed for random numbers", type=int, default=123456)

    args = parser.parse_args()

    # generate simulation
    rng = default_rng(args.seed)

    for nphotini in np.linspace(args.nphotini_min, args.nphotini_max, args.nphotini_nstep).astype(int):
        print(f'Creating {args.nsimulations} simulations with {nphotini} photons')
        for isimul in range(args.nsimulations):
            simul = VoigtComplexSimulation(
                name=args.name,
                nphotons=nphotini,
                fwhm_g=args.fwhm_g,
                xmin=args.xmin,
                xmax=args.xmax,
                nsampling=10001,
                rng=rng
            )

            hdr = fits.Header()
            hdr['complex'] = (args.name, 'name of the complex of lines')
            hdr['nphotini'] = (nphotini, 'initial number of photons')
            hdr['fwhm_g'] = (args.fwhm_g, 'Full Width at Half Maximum (Gaussian part)')
            hdr['xmin'] = (args.xmin, 'minimum valid energy (eV)')
            hdr['xmax'] = (args.xmax, 'maximum valid energy (eV)')
            hdr['seed'] = (args.seed, 'seed for random number generation')
            hdr['nphotsim'] = (simul.nphotons_simul, 'final number of valid photons')
            hdr['nleft'] = (simul.nleft, 'number of photons with energy below XMIN')
            hdr['nright'] = (simul.nright, 'number of photons with energy above XMAX')
            hdr['isimul'] = (isimul + 1, f'simulation number (out of {args.nsimulations})')

            primary_hdu = fits.PrimaryHDU(header=hdr)
            coldata = fits.Column(name='energy', format='D', array=simul.photons_simul)
            hdu = fits.BinTableHDU.from_columns([coldata])
            hdul = fits.HDUList([primary_hdu, hdu])
            outfname = f'Vsims_fwhm_{args.fwhm_g:.2f}_nphotini_{nphotini:05d}_isimul_{isimul+1:06d}.fits'
            hdul.writeto(outfname, overwrite=True)


if __name__ == "__main__":

    main()
