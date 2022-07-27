import numpy as np


class VoigtComplexDefinition:
    """Definition of a Voigt Complex.

    Parameters
    ----------
    name : str
        Name of the complex of lines.

    Attributes
    ----------
    name : str
        Name of the complex of lines.
    nvoigt : int
        Number of Voigt profiles in the complex.
    labels : list of str
        List with labels for each Voigt profile.
    energies_eV: numpy array
        Array with the energy of corresponding to the center of each line.
    global_mean : float
        Approximate mean energy of the line complex.
    fwhms_eV: numpy array
        Array with the FWHM associated to each line.
    rel_amplitudes : numpy array
        Array with relative amplitudes of each line.
    areas : numpy array
        Array with area under each line.

    """

    def __init__(self, name):
        if name == "MnKa":
            # MnKa data from:
            # https://heasarc.gsfc.nasa.gov/docs/hitomi/calib/caldb_doc/asth_sxs_caldb_linefit_v20161223.pdf
            self.name = name
            self.nvoigt = 8
            self.labels = ["Ka11", "Ka12",  "Ka13",  "Ka14", "Ka15", "Ka16",
                           "Ka21", "Ka22"]
            self.energies_eV = np.array(
                [5898.882, 5897.898, 5894.864, 5896.566, 5899.444, 5902.712,
                 5887.772, 5886.528],
                dtype=np.float64
            )
            self.global_mean = 5898.0
            self.fwhms_eV = np.array(
                [1.7145, 2.0442, 4.4985, 2.6616, 0.97669, 1.5528,
                 2.3604, 4.2168],
                dtype=np.float64
            )
            self.rel_amplitudes = np.array(
                [0.784, 0.263, 0.067, 0.095, 0.071, 0.011, 0.369, 0.1],
                dtype=np.float64
            )
            self.areas = np.array(
                [0.3523, 0.1409, 0.07892, 0.06624, 0.01818, 0.004475,
                 0.2283, 0.1106],
                dtype=np.float64
            )
        else:
            raise SystemError(f"Invalid Voigt Complex name: {name}")
