from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore
from tqdm.auto import tqdm

DIR_MODELS = '/vol/aibn182/data1/vjadhav/models_and_tools/models/'
# DIR_MODELS = '/Users/vikrantjadhav/Documents/work/models_and_tools/models/'
if not os.path.exists(DIR_MODELS):
    raise FileNotFoundError(
        'Either provide path (DIR_MODELS) for the models folder from https://github.com/jikrant3/models_and_tools/tree/main/models or skip plotting isochrones.')


class Spectrum:
    """
    A class to represent a spectrum, including its wavelength, flux, and errors.

    Attributes:
        x (ndarray): Logarithmic wavelength values.
        y (ndarray): Logarithmic flux values.
        frac_flux_err (ndarray or float or None): Fractional flux error.
        seed (int): Seed for random noise generation.

    Methods:
        wavelength(): Returns the wavelength in Angstroms.
        flux(): Returns the flux in erg/s/Angstrom.
        flux_err(): Returns the flux error based on fractional flux error.
        error_zero(): Checks if the error is zero.
        __add__(spec2): Adds two spectra and returns a new spectrum.
        __sub__(spec2): Subtracts two spectra and returns a new spectrum.
        __truediv__(spec2): Divides two spectra and returns a new spectrum.
        add_noise(σ, seed=0): Adds noise to the spectrum.
        plot(ax, **kwargs): Plots the spectrum.
        plot_physical(ax, **kwargs): Plots the spectrum in physical units.
    """

    def __init__(self,
                 x,
                 y,
                 frac_flux_err: None | float | np.ndarray = None,
                 seed: int = 0) -> None:
        """
        Initializes the Spectrum object.

        Args:
            x (ndarray): Logarithmic wavelength values.
            y (ndarray): Logarithmic flux values.
            frac_flux_err (ndarray, optional): Fractional flux error. Default is None.
            seed (int, optional): Seed for random noise generation. Default is 0.
        """
        self.x = x
        self.y = y
        self.add_noise(σ=frac_flux_err, seed=seed)

    @property
    def wavelength(self):
        """
        Converts logarithmic wavelength values to linear scale in Angstrom.

        Returns:
            Quantity: Wavelength in Angstrom.
        """
        return 10**self.x * u.Angstrom

    @property
    def flux(self):
        """
        Converts logarithmic flux values to linear scale in erg/s/Angstrom.

        Returns:
            Quantity: Flux in erg/s/Angstrom.
        """
        return 10**self.y * u.erg/u.s/u.Angstrom

    @property
    def flux_err(self):
        """
        Calculates the flux error based on fractional flux error.

        Returns:
            Quantity: Flux error in erg/s/Angstrom.
        """
        return self.frac_flux_err * self.flux

    @property
    def error_zero(self):
        """
        Checks if the fractional flux error is zero for all values.

        Returns:
            bool: True if all errors are zero, False otherwise.
        """
        return (np.sum(abs(self.frac_flux_err)) == 0)

    def __add__(self, spec2):
        """
        Adds two spectra and returns a new Spectrum object.

        Args:
            spec2 (Spectrum): The spectrum to be added.

        Returns:
            Spectrum: The resulting spectrum after addition.
        """
        flux_new = self.flux + spec2.flux
        flux_err_1 = self.flux_err
        flux_err_2 = spec2.flux_err
        flux_err = np.hypot(flux_err_1, flux_err_2)
        frac_flux_err = (flux_err/flux_new).value
        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_flux_err=frac_flux_err)

    def __sub__(self, spec2):
        """
        Subtracts two spectra and returns a new Spectrum object.

        Args:
            spec2 (Spectrum): The spectrum to be subtracted.

        Returns:
            Spectrum: The resulting spectrum after subtraction.
        """
        flux_new = self.flux - spec2.flux
        flux_err_1 = self.flux_err
        flux_err_2 = spec2.flux_err
        flux_err = np.hypot(flux_err_1, flux_err_2)
        frac_flux_err = (flux_err/flux_new).value
        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_flux_err=frac_flux_err)

    def __truediv__(self, spec2):
        """
        Divides two spectra and returns a new Spectrum object.

        Args:
            spec2 (Spectrum): The spectrum to divide by.

        Returns:
            Spectrum: The resulting spectrum after division.
        """
        flux_new = self.flux / spec2.flux
        frac_flux_err_1 = self.frac_flux_err
        frac_flux_err_2 = spec2.frac_flux_err
        frac_flux_err = np.hypot(frac_flux_err_1, frac_flux_err_2)

        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_flux_err=frac_flux_err)

    def add_noise(self, σ, seed=0):
        """
        Adds noise to the spectrum based on fractional flux error.

        Args:
            σ (None or float or ndarray): The fractional noise level or array of noise values.
            seed (int, optional): Seed for random noise generation. Default is 0.

        Raises:
            ValueError: If the noise array length does not match the data length.
        """
        if (σ is None) or (σ is 0) or (σ is 0.0):
            self.frac_flux_err = np.zeros(len(self.x))
            self.y_err = self.frac_flux_err/np.log(10)
            return

        if (isinstance(σ, list)) or (isinstance(σ, np.ndarray)):
            noise = []
            if len(σ) != len(self.y):
                raise ValueError(
                    'σ must be either float or a list similar to length of x.')
            for idx, sigma in enumerate(σ):
                rng = np.random.default_rng(seed+idx)
                noise.append(rng.normal(0, sigma, 1)[0])
            noise = np.array(noise)

        if (isinstance(σ, int)) or (isinstance(σ, float)):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, σ, len(self.y))

        noise = np.where(noise < -0.99, -0.99, noise)
        _flux = 10**self.y
        _flux = _flux*(1+noise)
        self.frac_flux_err = abs(noise)
        self.y = np.log10(_flux)
        self.y_err = self.frac_flux_err/np.log(10)

    def plot(self, ax=None, **kwargs):
        """
        Plots the spectrum using the original plotting method.

        Args:
            ax (matplotlib axis, optional): The axis to plot on. If None, a new axis is created.
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        Plotter.plot_spectrum_original(self, ax=ax, **kwargs)

    def plot_physical(self, ax=None, **kwargs):
        """
        Plots the spectrum in physical units using the original plotting method.

        Args:
            ax (matplotlib axis, optional): The axis to plot on. If None, a new axis is created.
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        Plotter.plot_spectrum_original_physical(self, ax=ax, **kwargs)


@u.quantity_input
def calc_luminosity(radius: u.m, T: u.Kelvin):
    """
    Calculate the luminosity of a star based on its radius and temperature using the Stefan-Boltzmann law.

    Args:
        radius (Quantity): The radius of the star in meters.
        T (Quantity): The temperature of the star in Kelvins.

    Returns:
        Quantity: The luminosity of the star in solar luminosities (Lsun).
    """
    L = (4 * np.pi * const.sigma_sb * radius**2 * T**4).to(u.solLum)
    return L.to(u.solLum)


@u.quantity_input
def calc_luminosity_from_T_sf_distance(T: u.Kelvin, sf, distance: u.pc):
    """
    Calculate luminosity based on temperature, scaling factor and distance.

    Args:
        T (Quantity): Temperature in Kelvins.
        sf (float): Scaling factor (dimensionless).
        distance (Quantity): Distance to the star in parsecs.

    Returns:
        Quantity: Luminosity in solar luminosities (Lsun).
    """
    radius = calc_radius_from_sf(sf, distance)
    return calc_luminosity(radius, T)


@u.quantity_input
def calc_sf(radius: u.m, distance: u.m):
    """
    Calculate the scaling factor for a given radius and distance.

    Args:
        radius (Quantity): The radius of the star in meters.
        distance (Quantity): The distance to the star in meters.

    Returns:
        Quantity: The scaling factor (dimensionless).
    """
    return ((radius/(distance))**2).decompose()


@u.quantity_input
def calc_radius_from_sf(sf, distance: u.m):
    """
    Calculate the radius of a star based on the scaling factor and distance.

    Args:
        sf (float): Scaling factor (dimensionless).
        distance (Quantity): Distance to the star in meters.

    Returns:
        Quantity: The radius of the star in solar radii (Rsun).
    """
    return (distance*(sf**0.5)).to(u.solRad)


@u.quantity_input
def calc_radius(L: u.solLum, T: u.Kelvin):
    """
    Calculate the radius of a star based on its luminosity and temperature.

    Args:
        L (Quantity): Luminosity of the star in solar luminosities (Lsun).
        T (Quantity): Temperature of the star in Kelvins.

    Returns:
        Quantity: The radius of the star in solar radii (Rsun).
    """
    R2 = L / (4 * np.pi * const.sigma_sb * T**4)
    return (R2**0.5).to(u.solRad)


@u.quantity_input
def estimate_errors_Single(T: u.Kelvin,
                           L: u.solLum,
                           σ: None | float | np.ndarray,
                           x: np.ndarray,
                           niter=50,
                           name='',
                           plot=True,
                           plot_name=None):
    """
    Estimate the errors for fitting a single star model.

    Args:
        T (Quantity): Temperature of the star in Kelvins.
        L (Quantity): Luminosity of the star in solar luminosities (Lsun).
        σ (None or float or np.ndarray): Flux error or noise level.
        x (np.ndarray): log(wavelength [Angstrom]) .
        niter (int, optional): Number of iterations to run. Defaults to 50.
        name (str, optional): Name of the star. Defaults to ''.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plot_name (str, optional): Path to save the plot if provided. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the fitted parameters and errors for each iteration.
    """
    fit_params = []
    for seed in range(niter):
        A = Star(T=T, L=L, σ=σ, seed=seed, x=x, name=name)
        A.fit_bb_Single(use_priors=True)
        if hasattr(A, 'logT_Single'):
            _fit_params_Single = dict(name=A.name,
                                      seed=A.seed,
                                      logT=A.logT,
                                      logL=A.logL,
                                      logT_Single=A.logT_Single,
                                      logL_Single=A.logL_Single,
                                      sigma=σ)
        else:
            _fit_params_Single = dict(name=A.name,
                                      seed=A.seed,
                                      logT=A.logT,
                                      logL=A.logL,
                                      logT_Single=np.nan,
                                      logL_Single=np.nan,
                                      sigma=σ)
        fit_params.append(_fit_params_Single)
    df_fit_params = pd.DataFrame(fit_params)
    convergence_rate = (np.sum(np.isfinite(df_fit_params.logT_Single)))/niter

    df_summary = df_fit_params.median(numeric_only=True)
    std = df_fit_params.std(numeric_only=True)
    df_summary['T_Single'] = 10**df_summary['logT_Single']
    df_summary['L_Single'] = 10**df_summary['logL_Single']
    df_summary['e_T_Single'] = abs(
        std['logT_Single']*np.log(10)*df_summary['T_Single'])
    df_summary['e_L_Single'] = abs(
        std['logL_Single']*np.log(10)*df_summary['L_Single'])

    print('T_in  = %f\nL_in  = %f' % (T.value, L.value))
    print('T_fit = %f ± %f\nL_fit = %f ± %f' % (df_summary.T_Single, df_summary.e_T_Single,
                                                df_summary.L_Single, df_summary.e_L_Single))
    print('  logT= %f ± %f\n  logL= %f ± %f' % (df_summary.logT_Single, std.logT_Single,
                                                df_summary.logL_Single, std.logL_Single))
    print('Convergence rate:%.2f' % convergence_rate)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        Plotter.plot_isochrone_and_wd(ax=ax)
        ax.scatter(A.logT, A.logL, marker='s', edgecolors='k', facecolor='none',
                   label='A (%d K, %.4f L$_{\odot}$)' % (T.value, L.value))
        ax.errorbar(df_summary.logT_Single, df_summary.logL_Single,
                    xerr=std.logT_Single, yerr=std.logL_Single,
                    label='A$_{fit}$ (%d±%d K, %.4f±%.4f L$_{\odot}$)' % (df_summary['T_Single'], df_summary['e_T_Single'],
                                                                          df_summary['L_Single'], df_summary['e_L_Single']))
        ax.scatter(df_fit_params.logT_Single, df_fit_params.logL_Single,
                   marker='.', alpha=0.9, c='C0', s=1)
        ax.legend()
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()
    return df_fit_params


@u.quantity_input
def estimate_errors_Double(T_A: u.Kelvin,
                           L_A: u.solLum,
                           T_B: u.Kelvin,
                           L_B: u.solLum,
                           σ: None | float | np.ndarray,
                           x: np.ndarray,
                           niter: int = 50,
                           name: str = '',
                           plot: bool = True,
                           plot_name: None | str = None):
    """
    Estimate the errors for fitting a binary star system model.

    Args:
        T_A (Quantity): Temperature of star A in Kelvins.
        L_A (Quantity): Luminosity of star A in solar luminosities (Lsun).
        T_B (Quantity): Temperature of star B in Kelvins.
        L_B (Quantity): Luminosity of star B in solar luminosities (Lsun).
        σ (None or float or np.ndarray): Flux error or noise level.
        x (np.ndarray): log(wavelength [Angstrom]) .
        niter (int, optional): Number of iterations to run. Defaults to 50.
        name (str, optional): Name of the binary system. Defaults to ''.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plot_name (str, optional): Path to save the plot if provided. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the fitted parameters and errors for each iteration.
    """
    fit_params = []
    for seed in range(niter):
        AB = Binary(T_A=T_A, L_A=L_A, T_B=T_B, L_B=L_B,
                    σ=σ, seed=seed, x=x, name=name)
        AB.fit_bb_Double(use_priors=True)
        fit_params_Single = dict(name=AB.name,
                                 seed_A=AB.A.seed,
                                 seed_B=AB.B.seed,
                                 logT_A=AB.A.logT,
                                 logL_A=AB.A.logL,
                                 logT_B=AB.B.logT,
                                 logL_B=AB.B.logL)
        if hasattr(AB, 'T_A'):
            fit_params_Double = dict(logT_A_Double=AB.logT_A,
                                     logL_A_Double=AB.logL_A,
                                     logT_B_Double=AB.logT_B,
                                     logL_B_Double=AB.logL_B)
        else:
            fit_params_Double = dict(logT_A_Double=np.nan,
                                     logL_A_Double=np.nan,
                                     logT_B_Double=np.nan,
                                     logL_B_Double=np.nan)
        _fit_params = fit_params_Single | fit_params_Double
        fit_params.append(_fit_params)
    df_fit_params = pd.DataFrame(fit_params)
    df_summary = df_fit_params.median(numeric_only=True)
    std = df_fit_params.std(numeric_only=True)
    df_summary['T_A_Double'] = 10**df_summary['logT_A_Double']
    df_summary['L_A_Double'] = 10**df_summary['logL_A_Double']
    df_summary['T_B_Double'] = 10**df_summary['logT_B_Double']
    df_summary['L_B_Double'] = 10**df_summary['logL_B_Double']
    df_summary['e_T_A_Double'] = abs(
        std['logT_A_Double']*np.log(10)*df_summary['T_A_Double'])
    df_summary['e_L_A_Double'] = abs(
        std['logL_A_Double']*np.log(10)*df_summary['L_A_Double'])
    df_summary['e_T_B_Double'] = abs(
        std['logT_B_Double']*np.log(10)*df_summary['T_B_Double'])
    df_summary['e_L_B_Double'] = abs(
        std['logL_B_Double']*np.log(10)*df_summary['L_B_Double'])
    convergence_rate = (np.sum(np.isfinite(df_fit_params.logT_A_Double)))/niter
    print('T_in  = [%f]\t [%f]\nL_in  = [%f]\t [%f]' %
          (T_A.value, T_B.value, L_A.value, L_B.value))
    print('T_fit = [%f ± %f]\t[%f ± %f]' % (df_summary['T_A_Double'], df_summary['e_T_A_Double'],
                                            df_summary['T_B_Double'], df_summary['e_T_B_Double']))
    print('L_fit = [%f ± %f]\t[%f ± %f]' % (df_summary['L_A_Double'], df_summary['e_L_A_Double'],
                                            df_summary['L_B_Double'], df_summary['e_L_B_Double']))
    print('Convergence rate:%.2f' % convergence_rate)
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        Plotter.plot_isochrone_and_wd(ax=ax)
        ax.scatter(df_summary.logT_A, df_summary.logL_A, marker='s', edgecolors='C0', facecolor='none',
                   label='A (%d K, %.4f L$_{\odot}$)' % (10**df_summary.logT_A, 10**df_summary.logL_A))
        ax.scatter(df_summary.logT_B, df_summary.logL_B, marker='s', edgecolors='C1', facecolor='none',
                   label='B (%d K, %.4f L$_{\odot}$)' % (10**df_summary.logT_B, 10**df_summary.logL_B))
        ax.errorbar(df_summary.logT_A_Double, df_summary.logL_A_Double,
                    xerr=std.logT_A_Double, yerr=std.logL_A_Double, marker='.',
                    label='A$_{fit}$ (%d±%d K, %.4f±%.4f L$_{\odot}$)' % (df_summary['T_A_Double'], df_summary['e_T_A_Double'],
                                                                          df_summary['L_A_Double'], df_summary['e_L_A_Double']),
                    color='C0')
        ax.errorbar(df_summary.logT_B_Double, df_summary.logL_B_Double,
                    xerr=std.logT_B_Double, yerr=std.logL_B_Double, marker='.',
                    label='B$_{fit}$ (%d±%d K, %.4f±%.4f L$_{\odot}$)' % (df_summary['T_B_Double'], df_summary['e_T_B_Double'],
                                                                          df_summary['L_B_Double'], df_summary['e_L_B_Double']),
                    color='C1')
        ax.scatter(df_fit_params.logT_A_Double,
                   df_fit_params.logL_A_Double, marker='.', alpha=0.9, c='C0', s=1)
        ax.scatter(df_fit_params.logT_B_Double,
                   df_fit_params.logL_B_Double, marker='.', alpha=0.9, c='C1', s=1)
        ax.legend()
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()
    return df_fit_params


class Plotter:
    def plot_isochrone_and_wd(ax=None):
        """
        Plot isochrones and white dwarf cooling curves.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.

        Notes:
            - Uses Parsec Isochrones for log(age) of 7, 8, 9, and 10, with [M/H] = 0.
            - Bergeron model white dwarf cooling curves are plotted for masses of 0.2, 0.5, and 1.3, with DA spectral type.
        """
        if ax is None:
            fig, ax = plt.subplots()
        iso = pd.read_csv(DIR_MODELS+'master_isochrone.csv')
        iso_7 = iso[iso.logAge == 7]
        iso_8 = iso[iso.logAge == 8]
        iso_9 = iso[iso.logAge == 9]
        iso_10 = iso[iso.logAge == 10]
        Bergeron_WD = pd.read_csv(DIR_MODELS + 'master_Bergeron_WD.csv')
        WD_02 = Bergeron_WD[(Bergeron_WD.mass == 0.2) &
                            (Bergeron_WD.spectral_type == 'DA')]
        WD_05 = Bergeron_WD[(Bergeron_WD.mass == 0.5) &
                            (Bergeron_WD.spectral_type == 'DA')]
        WD_13 = Bergeron_WD[(Bergeron_WD.mass == 1.3) &
                            (Bergeron_WD.spectral_type == 'DA')]
        ax.plot((iso_7.logTe), (iso_7.logL), label='',
                c='0.5', lw=0.5, rasterized=True, zorder=1)
        ax.plot((iso_8.logTe), (iso_8.logL), label='',
                c='0.5', lw=0.5, rasterized=True, zorder=1)
        ax.plot((iso_9.logTe), (iso_9.logL), label='',
                c='0.5', lw=0.5, rasterized=True, zorder=1)
        ax.plot((iso_10.logTe), (iso_10.logL), label='',
                c='0.5', lw=0.5, rasterized=True, zorder=1)
        ax.plot(np.log10(WD_02.Teff), WD_02.logL, label='', c='0.5',
                ls=(0, (5, 10)), lw=0.5, rasterized=True, zorder=1)
        ax.plot(np.log10(WD_05.Teff), WD_05.logL, label='', c='0.5',
                ls=(0, (5, 10)), lw=0.5, rasterized=True, zorder=1)
        ax.plot(np.log10(WD_13.Teff), WD_13.logL, label='', c='0.5',
                ls=(0, (5, 10)), lw=0.5, rasterized=True, zorder=1)

        ax.invert_xaxis()
        ax.set(xlabel='log(T [K])', ylabel='log(L [L$_⊙$])')

    def plot_spectrum_original(spec, ax=None, mode=None, **kwargs):
        """
        Plot the original spectrum with error bars.

        Args:
            spec (Spectrum): Spectrum object containing the spectral data.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.
            mode (str, optional): Plotting mode, default is None.
            **kwargs: Additional keyword arguments for `ax.errorbar`.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(spec.x, spec.y, yerr=spec.y_err, **kwargs)
        ax.set(xlabel='log(wavelength [Å])',
               ylabel='log(f [erg/s/Å/cm$^2$])')
        for x in np.log10([3800, 7500, 25000]):
            ax.axvline(x, c='0.5', lw=0.5)

    def plot_spectrum_original_physical(spec, ax=None, **kwargs):
        """
        Plot the spectrum in physical units with a log-log scale.

        Args:
            spec (Spectrum): Spectrum object containing wavelength and flux data.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.
            **kwargs: Additional keyword arguments for `ax.errorbar`.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(spec.wavelength, spec.flux, yerr=spec.flux_err, **kwargs)
        ax.set(xlabel='λ [Å]',
               ylabel='f [erg/s/Å/cm$^2$]')
        ax.loglog()
        for x in [3800, 7500, 25000]:
            ax.axvline(x, c='0.5', lw=0.5)

    def plot_spectrum_Single(source, ax=None):
        """
        Plot the fitted spectrum for a single blackbody model.

        Args:
            source (Source): Source object containing the fitted single blackbody spectrum.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots()
        label = 'Fit (%d K, %.4f L$_⊙$)' % (
            source.T_Single.value, source.L_Single.value)
        source.spectrum_Single.plot(ax=ax, c='C2', label=label)

    def plot_spectrum_Double(source, ax=None):
        """
        Plot the fitted spectrum for a double blackbody model.

        Args:
            source (Source): Source object containing the fitted double blackbody spectrum.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots()
        source.spectrum_Double.plot(ax=ax, c='C2', label='Fit')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        label = 'A$_{Fit}$ %d K, %.4f L$_⊙$' % (
            source.T_A.value, source.L_A.value)
        source.spectrum_A.plot(ax=ax, c='C0', label=label, lw=0.5)
        label = 'B$_{Fit}$ %d K, %.4f L$_⊙$' % (
            source.T_B.value, source.L_B.value)
        source.spectrum_B.plot(ax=ax, c='C1', label=label, lw=0.5)

    def plot_fitted_spectra(source, mode, ax=None):
        """
        Plot the fitted spectra based on the mode ('Single' or 'Double').

        Args:
            source (Source): Source object containing the spectrum and fit results.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new subplot is created.

        Returns:
            None
        """
        if mode == 'Single':
            Plotter.plot_spectrum_Single(source, ax=ax)
        if mode == 'Double':
            Plotter.plot_spectrum_Double(source, ax=ax)

    def plot_log_residual(source, mode, ax):
        """
        Plot the log residuals between the observed and fitted spectra.

        Args:
            source (Source): Source object containing the residual data.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.

        Returns:
            None
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.log_residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.log_residual_Double)
        ax.set_ylabel('Δlog(f)')

    def plot_residual(source, mode, ax):
        """
        Plot the residuals between the observed and fitted spectra.

        Args:
            source (Source): Source object containing the residual data.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.

        Returns:
            None
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.residual_Double)
        ax.set_ylabel('Δf')

    def plot_fractional_residual(source, mode, ax):
        """
        Plot the fractional residuals between the observed and fitted spectra.

        Args:
            source (Source): Source object containing the fractional residual data.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.

        Returns:
            None
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.fractional_residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.fractional_residual_Double)
        ax.set_ylabel('Δf/f')

    def plot_log_ewr(source, mode, ax, cax=None):
        """
        Plot the logarithmic error weighted residual (EWR).

        Args:
            source (Source): Source object containing the log EWR data.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.
            cax (matplotlib.axes.Axes, optional): Matplotlib color axis for color bar.

        Returns:
            None
        """
        if not source.spectrum.error_zero:
            if mode == 'Single':
                _y = source.log_ewr_Single
            if mode == 'Double':
                _y = source.log_ewr_Double
            ax.plot(source.spectrum.x, _y)
            ax.set_ylabel(r'$\frac{Δlog(f)}{ε_{log(f)}}$', size=15)

    def plot_ewr(source, mode, ax, cax=None):
        """
        Plot the error weighted residual (EWR).

        Args:
            source (Source): Source object containing the EWR data.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.
            cax (matplotlib.axes.Axes, optional): Matplotlib color axis for color bar.

        Returns:
            None
        """
        if not source.spectrum.error_zero:
            if mode == 'Single':
                _y = source.ewr_Single
            if mode == 'Double':
                _y = source.ewr_Double
            ax.plot(source.spectrum.x, _y)
            p0 = ax.scatter(source.spectrum.x, _y,
                            c=_y, cmap=plt.get_cmap('RdYlGn', 5),
                            vmin=-source.threshold_detection * 5./3.,
                            vmax=source.threshold_detection * 5./3.)
            plt.colorbar(p0, cax=cax, label='ewr')
            ax.set_ylabel(r'$\frac{Δf}{ε_f}$', size=15)

    def plot_fitted(source, mode, axes=None):
        """
        Plot the fitted spectra and residuals for a source.

        Args:
            source (Source): Source object containing the spectral data and fit results.
            mode (str): Fitting mode, either 'Single' or 'Double'.
            axes (list, optional): List of axes to plot on. If None, new axes are created.

        Returns:
            None
        """
        if axes is None:
            fig, axes = plt.subplots(figsize=(6, 6), nrows=6)
            [axi.set_axis_off() for axi in axes.ravel()]
            axes[0] = fig.add_axes([0.10, 0.60, 0.85, 0.39])
            axes[1] = fig.add_axes([0.10, 0.50, 0.85, 0.10])
            axes[2] = fig.add_axes([0.10, 0.40, 0.85, 0.10])
            axes[3] = fig.add_axes([0.10, 0.30, 0.85, 0.10])
            axes[4] = fig.add_axes([0.10, 0.20, 0.85, 0.10])
            axes[5] = fig.add_axes([0.10, 0.10, 0.85, 0.10])
            cax = fig.add_axes([0.95, 0.10, 0.02, 0.50])
            plt.setp(axes[0].get_xticklabels(), visible=False)
        source.plot(ax=axes[0], label='Original')
        Plotter.plot_fitted_spectra(source, mode=mode, ax=axes[0])
        Plotter.plot_residual(source, mode=mode, ax=axes[1])
        Plotter.plot_log_residual(source, mode=mode, ax=axes[2])
        Plotter.plot_ewr(source, mode=mode, ax=axes[3], cax=cax)
        Plotter.plot_log_ewr(source, mode=mode, ax=axes[4], cax=cax)
        Plotter.plot_fractional_residual(source, mode=mode, ax=axes[5])
        axes[5].set_xlabel('log(λ [Å])')
        axes[0].legend()


class Fitter:
    @staticmethod
    def bb_Single(spec, p0=[5000., -20], source=None):
        """
        Fit a single blackbody model to the spectrum.

        Args:
            spec (Spectrum): Spectrum object containing the observed flux data.
            p0 (list, optional): Initial guess for the fit parameters [Temperature, Log Scaling Factor].
            source (Source, optional): Source object to store the fitted parameters.

        Returns:
            array: Optimized fit parameters.
        """
        try:
            popt, pcov = curve_fit(Fitter.get_logflux_bb_Single,
                                   spec.x, spec.y, p0=p0,
                                   bounds=([100, -50],
                                           [10_000_000, 0]))
            if source is not None:
                source.T_Single = popt[0] * u.K
                source.logsf_Single = popt[1]
                source.sf_Single = 10**popt[1]
                source.L_Single = calc_luminosity_from_T_sf_distance(
                    source.T_Single, source.sf_Single, source.D)
                source.logT_Single = np.log10(source.T_Single.value)
                source.logL_Single = np.log10(source.L_Single.value)
                source.R_Single = calc_radius(source.L_Single, source.T_Single)

                y_fit = Fitter.get_logflux_bb_Single(spec.x, popt[0], popt[1])
                source.spectrum_Single = Spectrum(source.spectrum.x, y_fit)
                source.residual_Single = (
                    source.spectrum.flux - source.spectrum_Single.flux)
                source.fractional_residual_Single = (
                    source.residual_Single/source.spectrum.flux).value
                source.log_residual_Single = (
                    source.spectrum.y - source.spectrum_Single.y)
                if not spec.error_zero:
                    source.log_ewr_Single = source.log_residual_Single/spec.y_err
                    source.ewr_Single = (
                        source.residual_Single/spec.flux_err).value
                    source.misfits_ewr_Single = np.sum(
                        np.abs(source.ewr_Single) >= source.threshold_detection)
            return popt

        except RuntimeError:
            return np.full(2, np.nan)

    @staticmethod
    def bb_Double(spec, p0=[5000., -20, 5000, -20], source=None):
        """
        Fit a double blackbody model to the spectrum.

        Args:
            spec (Spectrum): Spectrum object containing the observed flux data.
            p0 (list, optional): Initial guess for the fit parameters [Temperature1, Log Scaling Factor1, Temperature2, Log Scaling Factor2].
            source (Source, optional): Source object to store the fitted parameters.

        Returns:
            array: Optimized fit parameters.
        """
        try:
            popt, pcov = curve_fit(Fitter.get_logflux_bb_Double,
                                   spec.x, spec.y, p0=p0,
                                   bounds=([100, -50, 100, -50],
                                           [10_000_000, 0, 10_000_000, 0]))
            if source is not None:
                source.T_A = popt[0] * u.K
                source.logsf_A = popt[1]
                source.sf_A = 10**popt[1]
                source.L_A = calc_luminosity_from_T_sf_distance(
                    source.T_A, source.sf_A, source.D)
                source.logT_A = np.log10(source.T_A.value)
                source.logL_A = np.log10(source.L_A.value)
                source.R_A = calc_radius(source.L_A, source.T_A)

                source.T_B = popt[2] * u.K
                source.logsf_B = popt[3]
                source.sf_B = 10**popt[3]
                source.L_B = calc_luminosity_from_T_sf_distance(
                    source.T_B, source.sf_B, source.D)
                source.logT_B = np.log10(source.T_B.value)
                source.logL_B = np.log10(source.L_B.value)
                source.R_B = calc_radius(source.L_B, source.T_B)

                y_fit = Fitter.get_logflux_bb_Double(
                    spec.x, popt[0], popt[1], popt[2], popt[3])
                source.spectrum_Double = Spectrum(source.spectrum.x, y_fit)
                source.residual_Double = (
                    source.spectrum.flux - source.spectrum_Double.flux)
                source.fractional_residual_Double = (
                    source.residual_Double/source.spectrum.flux).value
                source.log_residual_Double = (
                    source.spectrum.y - source.spectrum_Double.y)
                if not spec.error_zero:
                    source.log_ewr_Double = source.log_residual_Double/spec.y_err
                    source.ewr_Double = (
                        source.residual_Double/spec.flux_err).value
                    source.misfits_ewr_Double = np.sum(
                        np.abs(source.ewr_Double) >= source.threshold_detection)

                y_A = Fitter.get_logflux_bb_Single(spec.x, popt[0], popt[1])
                source.spectrum_A = Spectrum(spec.x, y_A)
                y_B = Fitter.get_logflux_bb_Single(spec.x, popt[2], popt[3])
                source.spectrum_B = Spectrum(spec.x, y_B)
            return popt
        except RuntimeError:
            return np.full(4, np.nan)

    @staticmethod
    def get_logflux_bb_Single(x, T, log_scaling_factor):
        """
        Calculate the logarithmic flux for a single blackbody model.

        Args:
            x (float or list): Logarithm of the wavelength in Angstroms.
            T (float): Temperature of the blackbody in Kelvin.
            log_scaling_factor (float): Logarithmic scaling factor (related to radius/distance).

        Returns:
            float or list: Logarithm of the blackbody flux in ergs/cm^2/s/A.
        """
        c1 = 3.7417749e-5                    # =2*!dpi*h*c*c
        c2 = 1.4387687                       # =h*c/k

        wave = (10**x)/1.e8     # angstroms to cm

        bbflux = c1 / (wave**5 * (np.exp(c2/wave/T)-1.))
        bbflux = bbflux*1.e-8                # ergs/cm2/s/a

        y = np.log10(bbflux)+log_scaling_factor

        return y

    @staticmethod
    def get_logflux_bb_Double(x, T_1, logsf_1, T_2, logsf_2):
        """
        Calculate the logarithmic flux for a double blackbody model.

        Args:
            x (float or list): Logarithm of the wavelength in Angstroms.
            T_1 (float): Temperature of the first blackbody in Kelvin.
            logsf_1 (float): Logarithmic scaling factor for the first blackbody.
            T_2 (float): Temperature of the second blackbody in Kelvin.
            logsf_2 (float): Logarithmic scaling factor for the second blackbody.

        Returns:
            float or list: Logarithm of the combined blackbody flux in ergs/cm^2/s/A.
        """
        c1 = 3.7417749e-5                    # =2*!dpi*h*c*c
        c2 = 1.4387687                       # =h*c/k

        wave = (10**x)/1.e8     # angstroms to cm
        bbflux_1 = c1 / (wave**5 * (np.exp(c2/wave/T_1)-1.))
        bbflux_1 = bbflux_1*1.e-8                # ergs/cm2/s/a
        bbflux_2 = c1 / (wave**5 * (np.exp(c2/wave/T_2)-1.))
        bbflux_2 = bbflux_2*1.e-8                # ergs/cm2/s/a

        y_1 = np.log10(bbflux_1)+logsf_1
        y_2 = np.log10(bbflux_2)+logsf_2
        y = np.log10(10**y_1 + 10**y_2)
        return y


class Star:
    @u.quantity_input
    def __init__(self,
                 T: u.Kelvin,
                 L: u.solLum,
                 σ: None | float | np.ndarray = 0,
                 seed=0,
                 D=10 * u.pc,
                 threshold_detection=5,
                 x : np.ndarray = np.linspace(3.13, 4.70, 50),
                 name='') -> None:
        """
        Initialize a Star object.

        Args:
            T (u.Kelvin): Temperature of the star.
            L (u.solLum): Luminosity of the star.
            σ (None or float or list, optional): Standard deviation in flux. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 0.
            D (u.pc, optional): Distance to the star. Defaults to 10 parsecs.
            threshold_detection (float, optional): Detection threshold (*flux_std) to identify badly fitting filters. Defaults to 5.
            x (np.ndarray, optional): log(wavelengths [Angstrom]) for the spectrum. Defaults to np.linspace(3.13, 4.70, 50).
            name (str, optional): Name of the star. Defaults to an empty string.
        """
        self.T = T.to(u.K)
        self.L = L.to(u.solLum)
        self.D = D
        self.σ = σ
        self.seed = seed
        self.threshold_detection = threshold_detection
        self.x = x
        self.name = name

        self.R = calc_radius(self.L, self.T)
        self.sf = calc_sf(self.R, self.D)
        self.logT = np.log10(self.T.value)
        self.logL = np.log10(self.L.value)
        self.logsf = np.log10(self.sf.value)
        self.y_std = σ/np.log(10)
        self.get_spectrum()

    def get_spectrum(self):
        """Generate the spectrum of the star and create a Spectrum object."""
        self.y = Fitter.get_logflux_bb_Single(
            self.x, self.T.value, self.logsf)
        self.spectrum = Spectrum(x=self.x,
                                 y=self.y,
                                 frac_flux_err=self.σ,
                                 seed=self.seed)

    def fit_bb_Single(self, use_priors=False, **kwargs):
        """
        Fit a single blackbody model to the star's spectrum.

        Args:
            use_priors (bool, optional): Whether to use prior values for fitting. Defaults to False.
            **kwargs: Additional keyword arguments passed to the fitting function.
        """
        if use_priors:
            p0 = [self.T.value, self.logsf]
        elif 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = [5000, -20]
        Fitter.bb_Single(self.spectrum, source=self, p0=p0, **kwargs)

    def fit_bb_Double(self, **kwargs):
        """
        Fit a double blackbody model to the star's spectrum.

        Args:
            **kwargs: Additional keyword arguments passed to the fitting function.
        """
        Fitter.bb_Double(self.spectrum, source=self, **kwargs)

    def plot(self, **kwargs):
        """
        Plot the original spectrum of the star.

        Args:
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        Plotter.plot_spectrum_original(self.spectrum, c='0', ls='', marker='.',
                                       **kwargs)

    def plot_fitted(self, mode, axes=None, plot_name=None):
        """
        Plot the fitted spectrum.

        Args:
            mode: Mode for plotting.
            axes (optional): Matplotlib axes to plot on. Defaults to None.
            plot_name (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
        """
        Plotter.plot_fitted(self, mode, axes=None)
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()


class Binary:
    @u.quantity_input
    def __init__(self,
                 T_A: u.Kelvin,
                 T_B: u.Kelvin,
                 L_A: u.solLum,
                 L_B: u.solLum,
                 σ: None | float | np.ndarray = 0,
                 seed=0,
                 D=10 * u.pc,
                 threshold_detection=5,
                 x=np.linspace(3.13, 4.70, 50),
                 name='') -> None:
        """
        Initialize a Binary system with two stars.

        Args:
            T_A (u.Kelvin): Temperature of the first star.
            T_B (u.Kelvin): Temperature of the second star.
            L_A (u.solLum): Luminosity of the first star.
            L_B (u.solLum): Luminosity of the second star.
            σ (None or float or np.ndarray,optional): Standard deviation in flux. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 0.
            D (u.pc, optional): Distance to the binary system. Defaults to 10 parsecs.
            threshold_detection (float, optional): Detection threshold (*flux_std) to identify badly fitting filters. Defaults to 5.
            x (np.ndarray, optional): Wavelengths for the spectrum. Defaults to np.linspace(3.13, 4.70, 50).
            name (str, optional): Name of the binary system. Defaults to an empty string.
        """
        self.A = Star(T=T_A,
                      L=L_A,
                      D=D,
                      σ=0,
                      threshold_detection=threshold_detection,
                      x=x)
        self.B = Star(T=T_B,
                      L=L_B,
                      D=D,
                      σ=0,
                      threshold_detection=threshold_detection,
                      x=x)

        self.threshold_detection = threshold_detection
        self.D = D
        self.x = x
        self.name = name

        self.spectrum = self.A.spectrum + self.B.spectrum
        self.spectrum.add_noise(σ, seed=seed)
        self.spectrum_ratio = self.B.spectrum/self.A.spectrum

    def plot(self, ax=None, **kwargs):
        """
        Plot the spectrum of the binary system and its individual components.

        Args:
            ax (optional): Matplotlib axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        if ax is None:
            fig, ax = plt.subplots()
        Plotter.plot_spectrum_original(self.spectrum, c='0', label='Original',
                                       ax=ax, ls='', marker='.')
        label = 'A %d K, %.4f L$_⊙$' % (self.A.T.value, self.A.L.value)
        Plotter.plot_spectrum_original(
            self.A.spectrum, c='C0', alpha=0.5, label=label, ax=ax, ls='', marker='.')
        label = 'B %d K, %.4f L$_⊙$' % (self.B.T.value, self.B.L.value)
        Plotter.plot_spectrum_original(
            self.B.spectrum, c='C1', alpha=0.5, label=label, ax=ax, ls='', marker='.')
        ax.legend()

    def fit_bb_Single(self, **kwargs):
        """
        Fit a single blackbody model to the binary system's spectrum.

        Args:
            **kwargs: Additional keyword arguments passed to the fitting function.
        """
        Fitter.bb_Single(self.spectrum, source=self, **kwargs)

    def fit_bb_Double(self, use_priors=False, **kwargs):
        """
        Fit a double blackbody model to the binary system's spectrum.

        Args:
            use_priors (bool, optional): Whether to use prior values for fitting. Defaults to False.
            **kwargs: Additional keyword arguments passed to the fitting function.
        """
        if use_priors:
            p0 = [self.A.T.value, self.A.logsf,
                  self.B.T.value, self.B.logsf]
        if 'p0' in kwargs:
            p0 = kwargs['p0']
        if ('p0' not in kwargs) & (not use_priors):
            p0 = [5000, -20, 6000, -20]
        Fitter.bb_Double(self.spectrum, source=self, p0=p0, **kwargs)

    def plot_fitted(self, mode, axes=None, plot_name=None):
        """
        Plot the fitted spectrum of the binary system.

        Args:
            mode: Mode for plotting.
            axes (optional): Matplotlib axes to plot on. Defaults to None.
            plot_name (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
        """
        Plotter.plot_fitted(self, mode, axes=None)
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()


class Grid:
    def __init__(self,
                 T_A: u.Kelvin,
                 L_A: u.solLum,
                 niter=1,
                 σ : float | np.ndarray = 0.001,
                 D=10 * u.pc,
                 threshold_filters=3,
                 threshold_detection=5,
                 x=np.linspace(3.13, 4.70, 50),
                 name='',
                 logT_B_list=None,
                 logL_B_list=None):
        """
        Initializes the Grid with the given parameters.

        Args:
            T_A (u.Kelvin): Temperature of star A.
            L_A (u.solLum): Luminosity of star A.
            niter (int, optional): Number of iterations for fitting. Defaults to 1.
            σ (float or np.ndarray, optional): Standard deviation in flux. Must be greater than 0. Defaults to 0.001.
            D (u.pc, optional): Distance to the binary system. Defaults to 10 parsecs.
            threshold_filters (int, optional): Threshold for detection of badly fitting filters. Defaults to 3.
            threshold_detection (int, optional): Detection threshold to identify issues. Defaults to 5.
            x (np.ndarray, optional): log(Wavelength [Angstrom]) range for spectrum fitting. Defaults to np.linspace(3.13, 4.70, 50).
            name (str, optional): Name for the grid. Defaults to ''.
            logT_B_list (list, optional): List of log(temperatures [K]) for star B.
            logL_B_list (list, optional): List of log(luminosities [solLum]) for star B.

        Raises:
            NotImplementedError: If σ is set to 0.
        """
        if (σ is 0):
            raise NotImplemented(
                'σ=0 is trivial and not implemented. Provide σ>0.')

        self.T_A = T_A
        self.L_A = L_A
        self.σ = σ
        self.D = D
        self.niter = niter
        self.threshold_filters = threshold_filters
        self.threshold_detection = threshold_detection
        self.x = x
        self.name = name

        self.logT_A = np.log10(T_A.value)
        self.logL_A = np.log10(L_A.value)

        if len(logT_B_list) != len(logL_B_list):
            raise ValueError(
                'logT_B_list and logL_B_list should have same length.')
        self.n_B = len(logT_B_list)
        self.logT_B_list, self.logL_B_list = logT_B_list, logL_B_list

    def get_AB_fit(self, T_A, T_B, L_A, L_B, σ, seed, name):
        """
        Creates a `Binary` instance and performs a fit.

        Args:
            T_A (u.Kelvin): Temperature of star A.
            T_B (u.Kelvin): Temperature of star B.
            L_A (u.solLum): Luminosity of star A.
            L_B (u.solLum): Luminosity of star B.
            σ (float): Standard deviation in flux.
            seed (int): Random seed for fitting.
            name (str): Name for the binary fit.

        Returns:
            Binary: A `Binary` instance with fitted parameters.
        """
        AB = Binary(T_A, T_B, L_A, L_B, σ=σ,
                    seed=seed, name=name,
                    D=self.D, x=self.x,
                    threshold_detection=self.threshold_detection)

        AB.fit_bb_Single(p0=[AB.A.T.value, AB.A.logsf])

        if hasattr(AB, 'misfits_ewr_Single'):
            if AB.misfits_ewr_Single >= self.threshold_filters:
                AB.fit_bb_Double(use_priors=True)
        return AB

    def get_sed_params(self, AB):
        """
        Extracts SED parameters from the fitted `Binary` instance.

        Args:
            AB (Binary): The `Binary` instance to extract parameters from.

        Returns:
            pd.DataFrame: DataFrame containing SED parameters.
        """
        df_sed = pd.DataFrame()
        df_sed['x'] = AB.x
        df_sed['seed_A'] = AB.A.seed
        df_sed['seed_B'] = AB.B.seed
        df_sed['logT_B'] = AB.B.logT
        df_sed['logL_B'] = AB.B.logL
        df_sed['y'] = AB.spectrum.y
        df_sed['y_A'] = AB.A.spectrum.y
        df_sed['y_B'] = AB.B.spectrum.y
        df_sed['name'] = AB.name
        if hasattr(AB, 'logT_Single'):
            df_sed['residual_Single'] = AB.residual_Single.value
            df_sed['log_residual_Single'] = AB.log_residual_Single
            df_sed['ewr_Single'] = AB.ewr_Single
            df_sed['log_ewr_Single'] = AB.log_ewr_Single
            df_sed['fractional_residual_Single'] = AB.fractional_residual_Single

        if hasattr(AB, 'logT_A_Double'):
            df_sed['residual_Double'] = AB.residual_Double.value
            df_sed['log_residual_Double'] = AB.log_residual_Double
            df_sed['ewr_Double'] = AB.ewr_Double
            df_sed['log_ewr_Double'] = AB.log_ewr_Double
            df_sed['fractional_residual_Double'] = AB.fractional_residual_Double
        return df_sed

    def get_AB_fit_params(self, AB):
        """
        Retrieves fitting parameters from the `Binary` instance.

        Args:
            AB (Binary): The `Binary` instance to extract fit parameters from.

        Returns:
            dict: Dictionary containing fit parameters.
        """
        fit_params_priors = dict(name=AB.name,
                                 seed_A=AB.A.seed,
                                 seed_B=AB.B.seed,
                                 logT_A=AB.A.logT,
                                 logL_A=AB.A.logL,
                                 logT_B=AB.B.logT,
                                 logL_B=AB.B.logL)
        if hasattr(AB, 'logT_Single'):
            fit_params_Single = dict(logT_Single=AB.logT_Single,
                                     logL_Single=AB.logL_Single)
        else:
            fit_params_Single = dict(logT_Single=np.nan,
                                     logL_Single=np.nan)
        if hasattr(AB, 'logT_A'):
            fit_params_Double = dict(logT_A_Double=AB.logT_A,
                                     logL_A_Double=AB.logL_A,
                                     logT_B_Double=AB.logT_B,
                                     logL_B_Double=AB.logL_B)
        else:
            fit_params_Double = dict(logT_A_Double=np.nan,
                                     logL_A_Double=np.nan,
                                     logT_B_Double=np.nan,
                                     logL_B_Double=np.nan)
        fit_params = fit_params_priors | fit_params_Single
        fit_params = fit_params | fit_params_Double
        return fit_params

    def process_fit_params(self, fit_params):
        """
        Calculates median and stddev of the df_fit_params. 
        And provides limits on the fit parameters based on typical stellar parameters.
        """
        df_fit_params = pd.DataFrame(fit_params)
        df_fit_params['id'] = df_fit_params['name'].str.extract(
            '(\d+)').astype(int)
        for column in ['logT_Single', 'logT_A_Double', 'logT_B_Double']:
            df_fit_params[column] = np.where(
                df_fit_params[column] > 7, 7, df_fit_params[column])
        for column in ['logL_Single', 'logL_A_Double', 'logL_B_Double']:
            df_fit_params[column] = np.where(
                df_fit_params[column] > 15, 15, df_fit_params[column])
            df_fit_params[column] = np.where(
                df_fit_params[column] < -10, -10, df_fit_params[column])
        for column in ['logT_A', 'logL_A', 'logT_B', 'logL_B',
                       'logT_Single', 'logL_Single',
                       'logT_A_Double', 'logL_A_Double', 'logT_B_Double', 'logL_B_Double']:
            df_fit_params[column[3:]] = 10**df_fit_params[column]

        self.df_fit_params = df_fit_params
        _group = df_fit_params.drop(columns=['name']).groupby(by='id')

        df_fit_params_median = _group.median()
        df_fit_params_median['convergence_rate'] = _group.count()[
            'logT_A_Double']/self.niter
        self.df_fit_params_median = df_fit_params_median.reset_index()

        df_fit_params_std = _group.std()
        df_fit_params_std['convergence_rate'] = df_fit_params_median['convergence_rate']
        self.df_fit_params_std = df_fit_params_std.reset_index()

    def calculate_params(self, refit=False):
        """
        Calculates and saves fitting parameters and SEDs.

        Args:
            refit (bool, optional): If True, refits and recalculates parameters. Defaults to False.
        """
        if not os.path.exists('outputs/%s' % self.name):
            os.makedirs('outputs/%s' % self.name)
        fit_params_name = 'outputs/%s/fit_params.csv' % self.name
        fit_params_median_name = 'outputs/%s/fit_params_median.csv' % self.name
        fit_params_std_name = 'outputs/%s/fit_params_std.csv' % self.name
        sed_niter0_name = 'outputs/%s/sed_niter0.csv' % self.name

        # If the files exists, they will be read (depends on refit = True or False)
        if os.path.isfile(fit_params_name):
            if not refit:
                self.df_fit_params = pd.read_csv(fit_params_name)
                self.df_fit_params_median = pd.read_csv(fit_params_median_name)
                self.df_fit_params_std = pd.read_csv(fit_params_std_name)
                self.df_sed_niter0 = pd.read_csv(sed_niter0_name)
                return

        df_sed_niter0 = pd.DataFrame()
        fit_params = []

        for idx in tqdm(range(self.n_B), desc=self.name, leave=False):
            # for idx in (range(self.n_B)):
            _T = 10**self.logT_B_list[idx] * u.K
            _L = 10**self.logL_B_list[idx] * u.solLum
            for seed in range(self.niter):
                name = 'id%d_niter%d_' % (idx, seed)
                AB = self.get_AB_fit(self.T_A, _T, self.L_A, _L,
                                     self.σ, seed, name)
                fit_params.append(self.get_AB_fit_params(AB))
                if seed == 0:
                    df_sed = self.get_sed_params(AB)
                    if idx == 0:
                        df_sed_niter0 = df_sed
                    else:
                        df_sed_niter0 = pd.concat([df_sed_niter0, df_sed])

        df_sed_niter0['id'] = df_sed_niter0['name'].str.extract(
            '(\d+)').astype(int)
        self.df_sed_niter0 = df_sed_niter0

        self.process_fit_params(fit_params)

        self.df_fit_params.to_csv(fit_params_name, index=False)
        self.df_fit_params_median.to_csv(fit_params_median_name, index=False)
        self.df_fit_params_std.to_csv(fit_params_std_name, index=False)
        self.df_sed_niter0.to_csv(sed_niter0_name, index=False)

    def plot_skeleton(self, ax=None, isochrones=True, zorder=0):
        """
        Plot the skeleton of the data on the provided axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
            isochrones (bool, optional): If True, plot isochrones. Defaults to True.
            zorder (int, optional): The z-order for the scatter plot. Defaults to 0.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        if isochrones:
            Plotter.plot_isochrone_and_wd(ax=ax)
        ax.scatter(self.logT_A, self.logL_A, s=50, marker='s', zorder=zorder,
                   facecolors='none', edgecolors='k')
        ax.scatter(self.logT_B_list, self.logL_B_list, c='0.3',
                   zorder=-1, s=10, marker='.', rasterized=True)

    def plot_sed_patches(self, param, ax=None):
        """
        Plot the SED patches on the provided axis.

        Args:
            param (str): The parameter to color the SED patches.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.

        Returns:
            None
        """
        def x_to_logT(x, logT, delta_logT, x_new, buffer=4):
            """
            Perform a linear interpolation to estimate logT values based on a range of x values.

            Args:
                x (array-like): The original x values (independent variable).
                logT (float): The base logT value.
                delta_logT (float): The range of logT values.
                x_new (float): The new x value for which logT needs to be estimated.
                buffer (int, optional): A buffer to adjust interpolation sensitivity. Defaults to 4.

            Returns:
                float: The estimated logT value for the given x_new.
            """
            x1, y1 = x.min(), logT + delta_logT/buffer
            x2, y2 = x.max(), logT - delta_logT/buffer
            logT_new = y1 + (y2-y1) * (x_new-x1) / (x2-x1)
            return logT_new

        def y_to_logL(y, ymin, yptp, height=1.2, offset=0.2):
            """
            Convert y values to logL values based on a given height and offset.

            Args:
                y (array-like): The y values to be converted.
                ymin (float): The minimum y value.
                yptp (float): The peak-to-peak range of y values.
                height (float, optional): The scaling factor for the y values. Defaults to 1.2.
                offset (float, optional): The offset to adjust the y values. Defaults to 0.2.

            Returns:
                array-like: The converted logL values.
            """
            y = y - ymin
            y = height * ((1-offset) * y/yptp + offset)
            return y

        if ax is None:
            fig, ax = plt.subplots()

        delta_logT = self.logT_B_list[1]-self.logT_B_list[0]
        df_sed_niter0 = self.df_sed_niter0
        _logL_max = self.logL_B_list.max()
        for idx in range(self.n_B):
            _logT = self.logT_B_list[idx]
            _logL = self.logL_B_list[idx]
            df = df_sed_niter0[df_sed_niter0['id'] == idx].reset_index()
            df['x_patch'] = x_to_logT(self.x, _logT, delta_logT,
                                      self.x, buffer=4)
            df['y_patch'] = np.full(len(self.x), _logL)
            if param == 'ewr_Single':
                df.sort_values(by=param, key=abs, inplace=True)
            if np.any(np.isfinite(df[param].values)):
                p0 = ax.scatter(df.x_patch, df.y_patch, c=df[param], marker='|', zorder=2,
                                vmin=-self.threshold_detection*5./3., vmax=self.threshold_detection*5./3.,
                                cmap=plt.get_cmap('RdYlGn', 5))

            # Plotting actual SEDs
            if _logL == _logL_max:
                df_sed = self.df_sed_niter0
                df_sed = df_sed[df_sed['id'] == idx].reset_index()
                df = df.sort_values(by='x_patch').reset_index()
                x = df.x_patch.values[::-1]
                _filter = (df_sed.y_A > df_sed.y.min())
                y = y_to_logL(df_sed.y_A[_filter],
                              df_sed.y.min(), np.ptp(df_sed.y))
                ax.plot(x[_filter], _logL + y, lw=1, c='C0')
                _filter = (df_sed.y_B > df_sed.y.min())
                y = y_to_logL(df_sed.y_B[_filter],
                              df_sed.y.min(), np.ptp(df_sed.y))
                ax.plot(x[_filter], _logL + y, lw=1, c='C1')
                # y = y_to_logL(df_sed.y, df_sed.y.min(), np.ptp(df_sed.y))
                # ax.plot(x, _logL + y, c='C2', lw=0.5)
        plt.colorbar(p0, ax=ax, pad=0, label='EWR', extend='both')

    def plot_Double_fitting_lines(self, niter=None, ax=None):
        """
        Plot the double fitting lines on the provided axis.

        Args:
            niter (int, optional): The iteration number for the fitting. If None, median values are used.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if niter is None:
            df = self.df_fit_params_median
        else:
            df = self.df_fit_params
            df = df[df['name'].str.contains('_niter%d_' % niter)].reset_index()
        df_std = self.df_fit_params_std
        logT_B_list = df.logT_B.unique()
        logL_B_list = df.logL_B.unique()
        color_list = plt.cm.rainbow_r(np.linspace(
            0, 1, len(logT_B_list)))  # creates array of N colors
        good_fit = (self.df_fit_params_median.convergence_rate >= 0.5) & (
            df_std.logT_A_Double * np.log(10) <= 0.33)
        df_good = df[np.isfinite(df.logT_A_Double) & good_fit].reset_index()
        df_poor = df[np.isfinite(df.logT_A_Double) & ~good_fit].reset_index()

        ax.plot([df.logT_A, df.logT_A_Double], [df.logL_A,
                df.logL_A_Double], c='k', marker='.', ms=0.5)

        for idx, logT_B in enumerate(logT_B_list):
            for jdx, logL_B in enumerate(logL_B_list):
                _df = df_good[(df_good.logT_B == logT_B) &
                              (df_good.logL_B == logL_B)]
                ax.plot([_df.logT_B, _df.logT_B_Double],
                        [_df.logL_B, _df.logL_B_Double],
                        c=color_list[idx], ls='-',
                        marker='.', ms=1)
                _df = df_poor[(df_poor.logT_B == logT_B) &
                              (df_poor.logL_B == logL_B)]
                ax.plot([_df.logT_B, _df.logT_B_Double],
                        [_df.logL_B, _df.logL_B_Double],
                        c=color_list[idx], ls='--',
                        marker='.', ms=1)

    def plot_Double_fitting_points(self, ax=None, noisy=False):
        """
        Plot the double fitting points on the provided axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
            noisy (bool, optional): If True, plot noisy data points with varying sizes. Defaults to False.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        df = self.df_fit_params_median
        logT_B_list = np.sort(np.unique(df.logT_B))
        logL_B_list = np.sort(np.unique(df.logL_B))
        df_std = self.df_fit_params_std
        _filter = (np.isfinite(df.logT_A_Double)) &\
                  (df.convergence_rate >= 0.5) &\
                  (df_std.logT_A_Double * np.log(10) <= 0.33)
        df = df[_filter].reset_index()
        df_std = df_std[_filter].reset_index()

        color_list = plt.cm.rainbow_r(np.linspace(
            0, 1, len(logT_B_list)))  # creates array of N colors

        if noisy:
            df_noisy = self.df_fit_params
            df_noisy = df_noisy[np.isfinite(
                df_noisy.logT_A_Double)].reset_index()
            s_list = np.linspace(10, 100, len(logT_B_list))
            s_list = (df.logL_B - np.nanmin(df.logL_B))**2
            s_list = s_list/np.nanmax(s_list) * 90 + 10
            s_list = (df_noisy.logL_B - np.nanmin(df_noisy.logL_B))**2
            s_list = s_list/np.nanmax(s_list) * 90 + 10
            p0 = ax.scatter(df_noisy.logT_B_Double, df_noisy.logL_B_Double,
                            c=df_noisy.logT_B, cmap='rainbow_r', zorder=4, s=s_list,
                            marker='$◯$', lw=0.1, alpha=0.5)

        for idx, logTe_B in enumerate(logT_B_list):
            _df = df[(df.logT_B == logTe_B)]
            _df_std = df_std[(df.logT_B == logTe_B)]
            ax.plot([_df.logT_B, _df.logT_B_Double],
                    [_df.logL_B, _df.logL_B_Double],
                    c=color_list[idx], alpha=1, ls=':',
                    zorder=0)
            ax.errorbar(_df.logT_B_Double, _df.logL_B_Double,
                        xerr=_df_std.logT_B_Double, yerr=_df_std.logL_B_Double,
                        ls='', ecolor=color_list[idx], marker='d', color=color_list[idx],
                        lw=1)

    def plot_Double_fitting_std(self, ax=None, cax=None, colorbar=True, **kwargs):
        """
        Plot the double fitting standard deviations on the provided axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
            cax (matplotlib.axes.Axes, optional): The colorbar axes to use. If None, a new colorbar is created.
            colorbar (bool, optional): If True, include a colorbar. Defaults to True.
            **kwargs: Additional keyword arguments passed to `ax.scatter`.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        df = self.df_fit_params_median
        df_std = self.df_fit_params_std
        _filter = (np.isfinite(df.logT_A_Double)) &\
                  (df.convergence_rate >= 0.5) &\
                  (df_std.logT_A_Double * np.log(10) <= 0.33)
        df = df[_filter].reset_index()
        df_std = df_std[_filter].reset_index()

        cmap = plt.get_cmap('cool', 3)   # 3 discrete colors
        c = df_std.logT_B_Double * np.log(10) * 100
        p0 = ax.scatter(df.logT_B, df.logL_B, marker='d',
                        c=c, vmin=0, vmax=15, cmap=cmap, rasterized=True, **kwargs)
        if colorbar:
            plt.colorbar(p0, extend='max', cax=cax,
                         label='T$_{2,\,err}$ (%)')  # ε$_T$

    def plot(self, plot_name=None):
        """
        Generate a series of plots and save to a file if specified.

        Args:
            plot_name (str, optional): The path to save the plot. If None, the plot is not saved.

        Returns:
            None
        """
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 13))

        self.plot_skeleton(ax=ax[0, 0])
        self.plot_skeleton(ax=ax[0, 1])
        self.plot_skeleton(ax=ax[1, 0])
        self.plot_skeleton(ax=ax[1, 1])

        self.plot_sed_patches(ax=ax[0, 0], param='ewr_Single')
        self.plot_Double_fitting_lines(ax=ax[0, 1], niter=0)
        self.plot_Double_fitting_lines(ax=ax[1, 0])
        self.plot_Double_fitting_points(ax=ax[1, 1], noisy=False)
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()
