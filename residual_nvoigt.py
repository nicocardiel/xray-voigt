from ecdf_nvoigt import ecdf_nvoigt


def residual_nvoigt(params, x, vcd):
    """Compute differences between model and empirical CDF.

    Parameters
    ----------
    params : instance of lmfit.Parameters
        Parameters defining the set of Voigt profiles.
    x : numpy array
        Initial energies where the empirical and model CDF will be
        computed.
    vcd : VoigtComplexDefinition instance
        Instance of VoigtComplexDefinition.

    Returns
    -------
    result : numpy array
        Differences between model and empirical CDF computed at each
        value of the 'x' array.
    """

    xecdf_data, yecdf_data, ycdf_model = ecdf_nvoigt(x, vcd, params, plot=False)
    result = ycdf_model - yecdf_data
    return result
