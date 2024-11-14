from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from typing import Union

# DIR_MODELS = '/vol/aibn182/data1/vjadhav/models_and_tools/models/'
DIR_MODELS = '/Users/vikrantjadhav/Documents/work/models_and_tools/models/'
if not os.path.exists(DIR_MODELS):
    raise FileNotFoundError(
        'Either provide path (DIR_MODELS) for the models folder from https://github.com/jikrant3/models_and_tools/tree/main/models or skip plotting isochrones.')

TMIN, TMAX = 1_000, 1_000_000
logSFMIN, logSFMAX = -40., -10.
logLMIN, logLMAX = -10., 12.


@u.quantity_input
def R_T_to_L(radius: u.m, T: u.Kelvin) -> u.Quantity:
    """
    Calculate the luminosity of a star based on its radius and temperature.

    Uses the Stefan-Boltzmann law to determine the luminosity.

    Parameters
    ----------
    radius : u.Quantity
        The radius of the star in meters.
    T : u.Quantity
        The temperature of the star in Kelvins.

    Returns
    -------
    u.Quantity
        The luminosity of the star in solar luminosities (Lsun).

    Examples
    --------
    >>> radius = 1 * u.Rsun
    >>> T = 5800 * u.K
    >>> R_T_to_L(radius, T)
    <Quantity 1.019... solLum>
    """
    L = (4 * np.pi * const.sigma_sb * radius**2 * T**4).to(u.solLum)
    return L.to(u.solLum)


@u.quantity_input
def T_sf_distance_to_L(T: u.Kelvin, sf: float, distance: u.pc) -> u.Quantity:
    """
    Calculate luminosity based on temperature, scaling factor, and distance.

    Parameters
    ----------
    T : u.Quantity
        Temperature in Kelvins.
    sf : float
        Scaling factor (dimensionless).
    distance : u.Quantity
        Distance to the star in parsecs.

    Returns
    -------
    u.Quantity
        Luminosity in solar luminosities (Lsun).

    Examples
    --------
    >>> T = 5800 * u.K
    >>> sf = 1e-17
    >>> distance = 10 * u.pc
    >>> T_sf_distance_to_L(T, sf, distance)
    <Quantity 2.0056... solLum>
    """
    radius = sf_distance_to_R(sf, distance)
    return R_T_to_L(radius, T)


@u.quantity_input
def R_distance_to_sf(radius: u.m, distance: u.m) -> u.Quantity:
    """
    Calculate the scaling factor for a given radius and distance.

    Parameters
    ----------
    radius : u.Quantity
        The radius of the star in meters.
    distance : u.Quantity
        The distance to the star in meters.

    Returns
    -------
    float
        The scaling factor (dimensionless).

    Examples
    --------
    >>> radius = 1 * u.Rsun
    >>> distance = 10 * u.pc
    >>> R_distance_to_sf(radius, distance)
    5.083...e-18
    """
    return ((radius/(distance))**2).decompose()


@u.quantity_input
def sf_distance_to_R(sf: float, distance: u.m) -> u.Quantity:
    """
    Calculate the radius of a star based on the scaling factor and distance.

    Parameters
    ----------
    sf : float
        Scaling factor (dimensionless).
    distance : u.Quantity
        Distance to the star in meters.

    Returns
    -------
    u.Quantity
        The radius of the star in solar radii (Rsun).

    Examples
    --------
    >>> sf = 1e-17
    >>> distance = 10 * u.pc
    >>> sf_distance_to_R(sf, distance)
    <Quantity 1.4025... solRad>
    """
    return (distance*(sf**0.5)).to(u.solRad)


@u.quantity_input
def L_T_to_R(L: u.solLum, T: u.Kelvin) -> u.Quantity:
    """
    Calculate the radius of a star based on its luminosity and temperature.

    Parameters
    ----------
    L : u.Quantity
        Luminosity of the star in solar luminosities (Lsun).
    T : u.Quantity
        Temperature of the star in Kelvins.

    Returns
    -------
    u.Quantity
        The radius of the star in solar radii (Rsun).

    Examples
    --------
    >>> L = 1 * u.solLum
    >>> T = 5800 * u.K
    >>> L_T_to_R(L, T)
    <Quantity 0.990... solRad>
    """
    R2 = L / (4 * np.pi * const.sigma_sb * T**4)
    return (R2**0.5).to(u.solRad)


class Spectrum:
    """
    A class to represent a spectrum, including its wavelength, flux, and errors.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        frac_err: Union[float, np.ndarray, list] = 0.0,
        seed: int = 0
    ) -> None:
        """
        Initializes the Spectrum object.

        Parameters
        ----------
        x : np.ndarray
            Logarithmic wavelength [Angstrom] values.
        y : np.ndarray
            Logarithmic flux [erg/s/Angstrom] values.
        frac_err : float or np.ndarray or list, optional
            Fractional flux error. Default is 0.
        seed : int, optional
            Seed for random noise generation. Default is 0.
        """
        self.x = x
        self.y = y
        self.add_noise(frac_err=frac_err, seed=seed)

    @property
    def wavelength(self) -> u.Quantity:
        """
        Convert logarithmic wavelength values to linear scale in Angstroms.

        Returns
        -------
        u.Quantity
            Wavelength in Angstroms.
        """
        return 10**self.x * u.Angstrom

    @property
    def flux(self) -> u.Quantity:
        """
        Convert logarithmic flux values to linear scale in erg/s/Angstrom.

        Returns
        -------
        u.Quantity
            Flux in erg/s/Angstrom.
        """
        return 10**self.y * u.erg/u.s/u.Angstrom

    @property
    def flux_err(self) -> u.Quantity:
        """
        Calculate the flux error based on fractional flux error.

        Returns
        -------
        u.Quantity
            Flux error in erg/s/Angstrom.
        """
        return self.frac_flux_err * self.flux

    @property
    def error_zero(self) -> bool:
        """
        Check if the fractional flux error is zero for all values.

        Returns
        -------
        bool
            True if all errors are zero, False otherwise.
        """
        return (np.sum(abs(self.frac_flux_err)) == 0)

    def __add__(self, spec2: 'Spectrum') -> 'Spectrum':
        """
        Add two spectra and return a new Spectrum object.

        Parameters
        ----------
        spec2 : Spectrum
            The spectrum to be added.

        Returns
        -------
        Spectrum
            The resulting spectrum after addition.
        """
        flux_new = self.flux + spec2.flux
        flux_err_1 = self.flux_err
        flux_err_2 = spec2.flux_err
        flux_err = np.hypot(flux_err_1, flux_err_2)
        frac_flux_err = (flux_err/flux_new).value
        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_err=frac_flux_err)

    def __sub__(self, spec2: 'Spectrum') -> 'Spectrum':
        """
        Subtract two spectra and return a new Spectrum object.

        Parameters
        ----------
        spec2 : Spectrum
            The spectrum to be subtracted.

        Returns
        -------
        Spectrum
            The resulting spectrum after subtraction.

        Raises
        ------
        ValueError
            If the subtraction leads to negative flux.
        """
        flux_new = self.flux - spec2.flux
        if any(flux_new < 0):
            raise ValueError(
                'Subtraction not allowed because flux sutraction leading to negative flux.')
        flux_err_1 = self.flux_err
        flux_err_2 = spec2.flux_err
        flux_err = np.hypot(flux_err_1, flux_err_2)
        frac_flux_err = (flux_err/flux_new).value
        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_err=frac_flux_err)

    def __truediv__(self, spec2: 'Spectrum') -> 'Spectrum':
        """
        Divide two spectra and return a new Spectrum object.

        Parameters
        ----------
        spec2 : Spectrum
            The spectrum to divide by.

        Returns
        -------
        Spectrum
            The resulting spectrum after division.

        Note
        ----
        The units of flux1/flux2 are not normalized.
        """
        flux_new = self.flux / spec2.flux
        frac_flux_err_1 = self.frac_flux_err
        frac_flux_err_2 = spec2.frac_flux_err
        frac_flux_err = np.hypot(frac_flux_err_1, frac_flux_err_2)

        return Spectrum(self.x, np.log10(flux_new.value),
                        frac_err=frac_flux_err)

    def add_noise(self,
                  frac_err: Union[float, np.ndarray, list],
                  seed: int = 0
                  ) -> None:
        """
        Add noise to the spectrum based on fractional flux error.

        Parameters
        ----------
        frac_err : float or np.ndarray or list
            The fractional noise level or array of noise values.
        seed : int, optional
            Seed for random noise generation. Default is 0.

        Raises
        ------
        ValueError
            If the noise array length does not match the data length or is not float/list/np.array.
        """
        if (frac_err is 0) or (frac_err is 0.0):
            self.frac_flux_err = np.zeros(len(self.x))
            self.y_err = self.frac_flux_err/np.log(10)
            return

        elif (isinstance(frac_err, list)) or (isinstance(frac_err, np.ndarray)):
            noise = []
            if len(frac_err) != len(self.y):
                raise ValueError(
                    'frac_err (%s) must be either float. Or a list/np.array with same length as x.' % frac_err)
            for idx, _frac_err in enumerate(frac_err):
                rng = np.random.default_rng(seed+idx)
                noise.append(rng.normal(0, _frac_err, 1)[0])
            noise = np.array(noise)

        elif (isinstance(frac_err, int)) or (isinstance(frac_err, float)):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, frac_err, len(self.y))
        else:
            raise ValueError(
                'frac_err (%s) must be either float or list or np.array.' % frac_err)
        noise = np.where(noise < -0.99, -0.99, noise)
        _flux = 10**self.y
        _flux = _flux*(1+noise)
        self.frac_flux_err = abs(noise)
        self.y = np.log10(_flux)
        self.y_err = self.frac_flux_err/np.log(10)

    def plot(self, ax=None, **kwargs) -> None:
        """
        Plot the spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None, a new axis is created.
        **kwargs : dict
            Additional keyword arguments for the plotting function.
        """
        Plotter.plot_spectrum_original(self, ax=ax, **kwargs)

    def plot_physical(self, ax=None, **kwargs) -> None:
        """
        Plot the spectrum in physical units.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None, a new axis is created.
        **kwargs : dict
            Additional keyword arguments for the plotting function.
        """
        Plotter.plot_spectrum_original_physical(self, ax=ax, **kwargs)


class Plotter:
    @staticmethod
    def plot_isochrone_and_wd(ax=None) -> None:
        """
        Plot isochrones and white dwarf cooling curves.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.

        Notes
        -----
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
                ls=(0, (5, 10)), lw=0.9, rasterized=True, zorder=1)
        ax.plot(np.log10(WD_05.Teff), WD_05.logL, label='', c='0.5',
                ls=(0, (5, 10)), lw=0.9, rasterized=True, zorder=1)
        ax.plot(np.log10(WD_13.Teff), WD_13.logL, label='', c='0.5',
                ls=(0, (5, 10)), lw=0.9, rasterized=True, zorder=1)

        ax.invert_xaxis()
        ax.set(xlabel='log(T [K])', ylabel='log(L [L$_⊙$])')

    @staticmethod
    def plot_spectrum_original(spec: 'Spectrum',
                               ax=None,
                               plot_name: str = None,
                               **kwargs
                               ) -> None:
        """
        Plot the original spectrum with error bars.

        Parameters
        ----------
        spec : Spectrum
            Spectrum object containing the spectral data.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        **kwargs
            Additional keyword arguments for `ax.errorbar`.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(spec.x, spec.y, yerr=spec.y_err, **kwargs)
        ax.set(xlabel='log(wavelength [Å])',
               ylabel='log(f [erg/s/Å/cm$^2$])')
        for x in np.log10([3800, 7500, 25000]):
            ax.axvline(x, c='0.5', lw=0.5)
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_spectrum_original_physical(spec: 'Spectrum',
                                        ax=None,
                                        plot_name: str = None,
                                        **kwargs
                                        ) -> None:
        """
        Plot the spectrum in physical units with a log-log scale.

        Parameters
        ----------
        spec : Spectrum
            Spectrum object containing wavelength and flux data.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        **kwargs
            Additional keyword arguments for `ax.errorbar`.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(spec.wavelength, spec.flux, yerr=spec.flux_err, **kwargs)
        ax.set(xlabel='λ [Å]',
               ylabel='f [erg/s/Å/cm$^2$]')
        ax.loglog()
        for x in [3800, 7500, 25000]:
            ax.axvline(x, c='0.5', lw=0.5)
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_spectrum_Single(source,
                             ax=None,
                             plot_name: str = None
                             ) -> None:
        """
        Plot the fitted spectrum for a single blackbody model.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the fitted single blackbody spectrum.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        label = 'Fit (%d K, %.4f L$_⊙$)' % (
            source.T_Single.value, source.L_Single.value)
        source.spectrum_Single.plot(ax=ax, c='C2', label=label)
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_spectrum_Double(source,
                             ax=None,
                             plot_name: str = None
                             ) -> None:
        """
        Plot the fitted spectrum for a double blackbody model.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the fitted double blackbody spectrum.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
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
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_fitted_spectra(source,
                            mode: str,
                            ax=None,
                            plot_name: str = None
                            ) -> None:
        """
        Plot the fitted spectra based on the mode ('Single' or 'Double').

        Parameters
        ----------
        source : Star or Binary
            Source object containing the spectrum and fit results.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        """
        if mode == 'Single':
            Plotter.plot_spectrum_Single(source, ax=ax)
        if mode == 'Double':
            Plotter.plot_spectrum_Double(source, ax=ax)
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_log_residual(source,
                          mode: str,
                          ax
                          ) -> None:
        """
        Plot the log residuals between the observed and fitted spectra.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the residual data.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.log_residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.log_residual_Double)
        ax.set_ylabel('Δlog(f)')

    @staticmethod
    def plot_residual(source,
                      mode: str,
                      ax
                      ) -> None:
        """
        Plot the residuals between the observed and fitted spectra.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the residual data.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.residual_Double)
        ax.set_ylabel('Δf')

    @staticmethod
    def plot_fractional_residual(source,
                                 mode: str,
                                 ax
                                 ) -> None:
        """
        Plot the fractional residuals between the observed and fitted spectra.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the fractional residual data.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        """
        if mode == 'Single':
            ax.plot(source.spectrum.x, source.fractional_residual_Single)
        if mode == 'Double':
            ax.plot(source.spectrum.x, source.fractional_residual_Double)
        ax.set_ylabel('Δf/f')

    @staticmethod
    def plot_log_ewr(source,
                     mode: str,
                     ax
                     ) -> None:
        """
        Plot the logarithmic error weighted residual (EWR).

        Parameters
        ----------
        source : Star or Binary
            Source object containing the log EWR data.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        cax : matplotlib.axes.Axes, optional
            Matplotlib color axis for color bar.
        """
        if not source.spectrum.error_zero:
            if mode == 'Single':
                _y = source.log_ewr_Single
            if mode == 'Double':
                _y = source.log_ewr_Double
            ax.plot(source.spectrum.x, _y)
            ax.set_ylabel(r'$\frac{Δlog(f)}{ε_{log(f)}}$', size=15)

    @staticmethod
    def plot_ewr(source,
                 mode: str,
                 ax,
                 cax=None
                 ) -> None:
        """
        Plot the error weighted residual (EWR).

        Parameters
        ----------
        source : Star or Binary
            Source object containing the EWR data.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        cax : matplotlib.axes.Axes, optional
            Matplotlib color axis for color bar.
        """
        if not source.spectrum.error_zero:
            if mode == 'Single':
                _y = source.ewr_Single
            if mode == 'Double':
                _y = source.ewr_Double
            ax.plot(source.spectrum.x, _y)
            p0 = ax.scatter(source.spectrum.x, _y,
                            c=_y, cmap=plt.get_cmap('RdYlGn', 5),
                            vmin=-source.threshold_ewr * 5./3.,
                            vmax=source.threshold_ewr * 5./3.)
            plt.colorbar(p0, cax=cax, label='ewr')
            ax.set_ylabel(r'$\frac{Δf}{ε_f}$', size=15)

    @staticmethod
    def plot_fitted(source,
                    mode: str,
                    axes=None,
                    plot_name: str = None
                    ) -> None:
        """
        Plot the fitted spectra and residuals for a source.

        Parameters
        ----------
        source : Star or Binary
            Source object containing the spectral data and fit results.
        mode : str
            Fitting mode, either 'Single' or 'Double'.
        axes : list of matplotlib.axes.Axes, optional
            List of axes to plot on. If None, new axes are created.
        plot_name : str, optional
            File path to save the plot.
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
        Plotter.plot_log_ewr(source, mode=mode, ax=axes[4])
        Plotter.plot_fractional_residual(source, mode=mode, ax=axes[5])
        axes[5].set_xlabel('log(λ [Å])')
        axes[0].legend()
        Plotter.savefig(plot_name)

    @staticmethod
    def plot_error_estimate_Single(source,
                                   ax=None,
                                   plot_name: str = None
                                   ) -> None:
        """
        Plot error estimates for single blackbody fit parameters.

        Parameters
        ----------
        source : Star or Binary
            Source object containing parameter errors for the single blackbody fit.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        df_fit_params = source.df_error_fit_params
        good_fits = df_fit_params['flag_good_fit']
        df_fit_params_good = df_fit_params[good_fits]
        df_summary = source.df_error_summary

        ax.scatter(source.logT, source.logL, marker='s', edgecolors='C0', facecolor='none',
                   label='A (%d K, %.4f L$_{\odot}$)' % (source.T.value, source.L.value))

        x, y = df_summary.logT_Single_50, df_summary.logL_Single_50
        e_x_lower, e_x_upper = df_summary.e_logT_Single_lower, df_summary.e_logT_Single_upper
        e_y_lower, e_y_upper = df_summary.e_logL_Single_lower, df_summary.e_logL_Single_upper
        e_x = [[e_x_lower], [e_x_upper]]
        e_y = [[e_y_lower], [e_y_upper]]
        ax.errorbar(x, y, xerr=e_x, yerr=e_y, marker='.', color='C0',
                    label='A$_{fit}$ (%d$^{+%d}_{-%d}$ K, %.4f$^{+%.4f}_{-%.4f}$ L$_{\odot}$)' % (
                        10**x, df_summary.e_T_Single_upper, df_summary.e_T_Single_lower,
                        10**y, df_summary.e_L_Single_upper, df_summary.e_L_Single_lower))

        ax.scatter(df_fit_params_good.logT_Single, df_fit_params_good.logL_Single,
                   marker='.', alpha=0.9, c='C0', s=1, label='Individual fits (%d/%d)' % (sum(good_fits), len(good_fits)))
        ax.legend()
        ax.set_title(source.name + ' (convergence_rate=%.2f)' %
                     source.convergence_rate)

        Plotter.savefig(plot_name)

    @staticmethod
    def plot_error_estimate_Double(source,
                                   ax=None,
                                   plot_name: str = None
                                   ) -> None:
        """
        Plot error estimates for double blackbody fit parameters.

        Parameters
        ----------
        source : Star or Binary
            Source object containing parameter errors for the double blackbody fit.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        df_fit_params = source.df_error_fit_params
        good_fits = df_fit_params['flag_good_fit']
        df_fit_params_good = df_fit_params[good_fits]
        df_summary = source.df_error_summary
        ax.scatter(df_summary.logT_A, df_summary.logL_A, marker='s', edgecolors='C0', facecolor='none',
                   label='A (%d K, %.4f L$_{\odot}$)' % (source.A.T.value, source.A.L.value))
        ax.scatter(df_summary.logT_B, df_summary.logL_B, marker='s', edgecolors='C1', facecolor='none',
                   label='B (%d K, %.4f L$_{\odot}$)' % (source.B.T.value, source.B.L.value))

        x, y = df_summary.logT_A_Double_50, df_summary.logL_A_Double_50
        e_x_lower, e_x_upper = df_summary.e_logT_A_Double_lower, df_summary.e_logT_A_Double_upper
        e_y_lower, e_y_upper = df_summary.e_logL_A_Double_lower, df_summary.e_logL_A_Double_upper
        e_x = [[e_x_lower], [e_x_upper]]
        e_y = [[e_y_lower], [e_y_upper]]
        ax.errorbar(x, y, xerr=e_x, yerr=e_y, marker='.', color='C0',
                    label='A$_{fit}$ (%d$^{+%d}_{-%d}$ K, %.4f$^{+%.4f}_{-%.4f}$ L$_{\odot}$)' % (
                        10**x, df_summary.e_T_A_Double_upper, df_summary.e_T_A_Double_lower,
                        10**y, df_summary.e_L_A_Double_upper, df_summary.e_L_A_Double_lower))

        x, y = df_summary.logT_B_Double_50, df_summary.logL_B_Double_50
        e_x_lower, e_x_upper = df_summary.e_logT_B_Double_lower, df_summary.e_logT_B_Double_upper
        e_y_lower, e_y_upper = df_summary.e_logL_B_Double_lower, df_summary.e_logL_B_Double_upper
        e_x = [[e_x_lower], [e_x_upper]]
        e_y = [[e_y_lower], [e_y_upper]]
        ax.errorbar(x, y, xerr=e_x, yerr=e_y, marker='.', color='C1',
                    label='B$_{fit}$ (%d$^{+%d}_{-%d}$ K, %.4f$^{+%.4f}_{-%.4f}$ L$_{\odot}$)' % (
                        10**x, df_summary.e_T_B_Double_upper, df_summary.e_T_B_Double_lower,
                        10**y, df_summary.e_L_B_Double_upper, df_summary.e_L_B_Double_lower))

        ax.scatter(df_fit_params_good.logT_A_Double, df_fit_params_good.logL_A_Double,
                   marker='.', alpha=1, c='C0', s=5,
                   label='Individual fits (%d/%d)' % (sum(good_fits), len(good_fits)))
        ax.scatter(df_fit_params_good.logT_B_Double,
                   df_fit_params_good.logL_B_Double, marker='.', alpha=1, c='C1', s=5)
        ax.set_title(source.name + ' (convergence_rate=%.2f)' %
                     source.convergence_rate)
        ax.legend(fontsize=8, framealpha=0.3)

        Plotter.savefig(plot_name)

    @staticmethod
    def plot_pseudo_secondaries_Double(source,
                                       ax=None,
                                       plot_name: str = None
                                       ) -> None:
        """
        Plot the pseudo-secondaries of the inverse problem for a double blackbody fit.

        Parameters
        ----------
        source : Star or Binary
            Source object containing data for the inverse problem solution.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new subplot is created.
        plot_name : str, optional
            File path to save the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(source.B.logT, source.B.logL, s=50, marker='s', zorder=1,
                   facecolors='none', edgecolors='k')
        source.grid.plot_skeleton(ax=ax)
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        source.grid.plot_Double_fitting_lines(ax=ax, plot_poor=False)
        Plotter.savefig(plot_name)

    @staticmethod
    def savefig(plot_name: str = None) -> None:
        """
        Save the current plot to the specified file.

        Parameters
        ----------
        plot_name : str, optional
            The file name (including path) to save the plot.

        Notes
        -----
        Creates directories if they do not exist.
        """
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300,
                        bbox_inches='tight', facecolor='white')
            plt.close()


class Fitter:
    @staticmethod
    def bb_Single(spec: Spectrum,
                  p0: list[float] = [5000., -20],
                  source=None) -> np.ndarray:
        """
        Fit a single blackbody model to the spectrum.

        Parameters
        ----------
        spec : Spectrum
            Spectrum object containing the observed flux data.
        p0 : list of float, optional
            Initial guess for the fit parameters [Temperature, Log Scaling Factor] (default is [5000., -20]).
        source : Source, optional
            Source object to store the fitted parameters (default is None).

        Returns
        -------
        np.ndarray
            Optimized fit parameters as a NumPy array. If the fit fails or is rejected, returns an array filled with NaNs.
        """
        try:
            popt, pcov = curve_fit(Fitter.get_logflux_bb_Single,
                                   spec.x, spec.y, p0=p0,
                                   bounds=([TMIN, logSFMIN],
                                           [TMAX, logSFMAX]))
            # Rejecting near-boundry (1%) fits
            check_0 = np.isclose(popt[0], TMIN, rtol=0.01)
            check_1 = np.isclose(popt[0], TMAX, rtol=0.01)
            check_2 = np.isclose(popt[1], logSFMIN, rtol=0.01)
            check_3 = np.isclose(popt[1], logSFMAX, rtol=0.01)
            if np.sum([check_1, check_0, check_2, check_3]) > 0:
                return np.full(2, np.nan)

            if source is not None:
                # Rejecting unphysical Luminosity
                _logL = np.log10(T_sf_distance_to_L(T=popt[0]*u.K,
                                                    sf=10**popt[1],
                                                    distance=source.D).value)
                check_7 = ((_logL < logLMIN+0.01) | (_logL > logLMAX-0.01))
                if check_7:
                    return np.full(2, np.nan)

                source.T_Single = popt[0] * u.K
                source.logsf_Single = popt[1]
                source.sf_Single = 10**popt[1]
                source.L_Single = T_sf_distance_to_L(
                    source.T_Single, source.sf_Single, source.D)
                source.logT_Single = np.log10(source.T_Single.value)
                source.logL_Single = np.log10(source.L_Single.value)
                source.R_Single = L_T_to_R(source.L_Single, source.T_Single)

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
                        np.abs(source.ewr_Single) >= source.threshold_ewr)
            return popt

        except RuntimeError:
            return np.full(2, np.nan)

    @staticmethod
    def bb_Double(spec: Spectrum,
                  p0: list[float] = [5000., -20, 5000, -20],
                  source=None,
                  threshold_primary_match: float = None
                  ) -> np.ndarray:
        """
        Fit a double blackbody model to the spectrum.

        Parameters
        ----------
        spec : Spectrum
            Spectrum object containing the observed flux data.
        p0 : list of float, optional
            Initial guess for the fit parameters [Temperature1, Log Scaling Factor1, Temperature2, Log Scaling Factor2] (default is [5000., -20, 5000, -20]).
        source : Star or Binary, optional
            Source object to store the fitted parameters (default is None).
        threshold_primary_match : float, optional
            Threshold for matching the primary component, used for fit validation (default is None).

        Returns
        -------
        np.ndarray
            Optimized fit parameters as a NumPy array. If the fit fails or is rejected, returns an array filled with NaNs.
        """
        try:
            popt, pcov = curve_fit(Fitter.get_logflux_bb_Double,
                                   spec.x, spec.y, p0=p0,
                                   bounds=([TMIN, logSFMIN, TMIN, logSFMIN],
                                           [TMAX, logSFMAX, TMAX, logSFMAX]))

            # Rejecting near-boundry (1%) fits based on T and logsf
            check_0 = np.isclose([popt[0], popt[2]], TMIN, rtol=0.01)
            check_1 = np.isclose([popt[0], popt[2]], TMAX, rtol=0.01)
            check_2 = np.isclose([popt[1], popt[3]], logSFMIN, rtol=0.01)
            check_3 = np.isclose([popt[1], popt[3]], logSFMAX, rtol=0.01)
            if np.sum([check_1, check_0, check_2, check_3]) > 0:
                return np.full(4, np.nan)

            if source is not None:
                # Rejecting unphysical Luminosity
                _logL = np.log10(T_sf_distance_to_L(T=popt[0]*u.K,
                                                    sf=10**popt[1],
                                                    distance=source.D).value)
                check_6 = ((_logL < logLMIN+0.01) | (_logL > logLMAX-0.01))
                _logL = np.log10(T_sf_distance_to_L(T=popt[2]*u.K,
                                                    sf=10**popt[3],
                                                    distance=source.D).value)
                check_7 = ((_logL < logLMIN+0.01) | (_logL > logLMAX-0.01))
                if (check_6 | check_7):
                    return np.full(4, np.nan)

                # Rejecting bad primary
                if threshold_primary_match is not None:
                    check_4 = np.isclose(
                        popt[0], source.A.T.value, rtol=threshold_primary_match)
                    check_5 = np.isclose(
                        popt[1], source.A.logsf, atol=threshold_primary_match)
                    good_primary = (check_4 & check_5)
                    if not good_primary:
                        return np.full(4, np.nan)

                source.T_A = popt[0] * u.K
                source.logsf_A = popt[1]
                source.sf_A = 10**popt[1]
                source.L_A = T_sf_distance_to_L(
                    source.T_A, source.sf_A, source.D)
                source.logT_A = np.log10(source.T_A.value)
                source.logL_A = np.log10(source.L_A.value)
                source.R_A = L_T_to_R(source.L_A, source.T_A)

                source.T_B = popt[2] * u.K
                source.logsf_B = popt[3]
                source.sf_B = 10**popt[3]
                source.L_B = T_sf_distance_to_L(
                    source.T_B, source.sf_B, source.D)
                source.logT_B = np.log10(source.T_B.value)
                source.logL_B = np.log10(source.L_B.value)
                source.R_B = L_T_to_R(source.L_B, source.T_B)

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
                        np.abs(source.ewr_Double) >= source.threshold_ewr)

                y_A = Fitter.get_logflux_bb_Single(spec.x, popt[0], popt[1])
                source.spectrum_A = Spectrum(spec.x, y_A)
                y_B = Fitter.get_logflux_bb_Single(spec.x, popt[2], popt[3])
                source.spectrum_B = Spectrum(spec.x, y_B)
            return popt
        except RuntimeError:
            return np.full(4, np.nan)

    @staticmethod
    def get_logflux_bb_Single(x: Union[float, list[float], np.ndarray],
                              T: float,
                              log_scaling_factor: float
                              ) -> Union[float, np.ndarray]:
        """
        Calculate the logarithmic flux for a single blackbody model.

        Parameters
        ----------
        x : float or array-like
            Logarithm of the wavelength in Angstroms.
        T : float
            Temperature of the blackbody in Kelvin.
        log_scaling_factor : float
            Logarithmic scaling factor (related to radius/distance).

        Returns
        -------
        float or np.ndarray
            Logarithm of the blackbody flux in ergs/cm^2/s/Å.
        """
        c1 = 3.7417749e-5                    # =2*!dpi*h*c*c
        c2 = 1.4387687                       # =h*c/k

        wave = (10**x)/1.e8     # angstroms to cm

        bbflux = c1 / (wave**5 * (np.exp(c2/wave/T)-1.))
        bbflux = bbflux*1.e-8                # ergs/cm2/s/a

        y = np.log10(bbflux)+log_scaling_factor

        return y

    @staticmethod
    def get_logflux_bb_Double(x: Union[float, list[float], np.ndarray],
                              T_1: float,
                              logsf_1: float,
                              T_2: float,
                              logsf_2: float
                              ) -> Union[float, np.ndarray]:
        """
        Calculate the logarithmic flux for a double blackbody model.

        Parameters
        ----------
        x : float or array-like
            Logarithm of the wavelength in Angstroms.
        T_1 : float
            Temperature of the first blackbody in Kelvin.
        logsf_1 : float
            Logarithmic scaling factor for the first blackbody.
        T_2 : float
            Temperature of the second blackbody in Kelvin.
        logsf_2 : float
            Logarithmic scaling factor for the second blackbody.

        Returns
        -------
        float or np.ndarray
            Logarithm of the combined blackbody flux in ergs/cm^2/s/Å.
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
                 T: u.Quantity,
                 L: u.Quantity,
                 frac_err: Union[float, np.ndarray, list] = 0,
                 seed: int = 0,
                 D: u.Quantity = 10 * u.pc,
                 threshold_ewr: float = 5,
                 x: np.ndarray = np.linspace(3.1, 4.7, 9),
                 name: str = '') -> None:
        """
        Initialize a Star object.

        Parameters
        ----------
        T : u.Quantity
            Temperature of the star in Kelvin.
        L : u.Quantity
            Luminosity of the star in solar luminosities.
        frac_err : loat, or np.ndarray or list, optional
            Fractional error (also called sigma) in flux. Defaults to 0.
        seed : int, optional
            Random seed for reproducibility. Defaults to 0.
        D : u.Quantity, optional
            Distance to the star in parsecs. Defaults to 10 pc.
        threshold_ewr : float, optional
            Detection threshold for EWR to identify poorly fitting filters (with |EWR|>threshold_ewr). Defaults to 5.
        x : np.ndarray, optional
            Logarithmic wavelengths in Angstroms for the spectrum/SED. Pivot wavelengths of the filters. Defaults to `np.linspace(3.1, 4.7, 9)`.
        name : str, optional
            Name of the star. Defaults to an empty string.
        """
        self.T = T.to(u.K)
        self.L = L.to(u.solLum)
        self.D = D
        self.frac_err = frac_err
        self.seed = seed
        self.threshold_ewr = threshold_ewr
        self.x = x
        self.name = name

        self.R = L_T_to_R(self.L, self.T)
        self.sf = R_distance_to_sf(self.R, self.D)
        self.logT = np.log10(self.T.value)
        self.logL = np.log10(self.L.value)
        self.logsf = np.log10(self.sf.value)
        self.y_std = frac_err/np.log(10)
        self.get_spectrum()

    def get_spectrum(self) -> None:
        """
        Generate the spectrum of the star and create a Spectrum object.
        """
        self.y = Fitter.get_logflux_bb_Single(
            self.x, self.T.value, self.logsf)
        self.spectrum = Spectrum(x=self.x,
                                 y=self.y,
                                 frac_err=self.frac_err,
                                 seed=self.seed)

    def fit_bb_Single(self, use_priors: bool = False, **kwargs) -> None:
        """
        Fit a single blackbody model to the star's spectrum.

        Parameters
        ----------
        use_priors : bool, optional
            If True, use prior values for fitting. Defaults to False.
        **kwargs
            Additional keyword arguments passed to the fitting function.
        """
        if use_priors:
            p0 = [self.T.value, self.logsf]
        elif 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = [5000, -20]
        Fitter.bb_Single(self.spectrum, source=self, p0=p0, **kwargs)

    def fit_bb_Double(self, **kwargs) -> None:
        """
        Fit a double blackbody model to the star's spectrum.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the fitting function.
        """
        Fitter.bb_Double(self.spectrum, source=self, **kwargs)

    def plot(self, **kwargs) -> None:
        """
        Plot the original spectrum of the star.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the plotting function.
        """
        Plotter.plot_spectrum_original(self.spectrum, c='0', ls='', marker='.',
                                       **kwargs)

    def plot_fitted(self, mode: str, axes=None, plot_name: str = None) -> None:
        """
        Plot the fitted spectrum.

        Parameters
        ----------
        mode : str
            Mode for plotting.
        axes : plt.Axes, optional
            Matplotlib axes to plot on. Defaults to None.
        plot_name : str, optional
            Path to save the plot. If None, the plot is not saved. Defaults to None.
        """
        Plotter.plot_fitted(self, mode, axes=None)
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()

    def estimate_errors(self,
                        niter: int = 100,
                        verbose: bool = True
                        ) -> None:
        """
        Estimate the fitting errors using Monte Carlo simulations.

        Parameters
        ----------
        niter : int, optional
            Number of iterations for error estimation. Defaults to 100.
        verbose : bool, optional
            If True, prints summary details. Defaults to True.
        """
        fit_params = []
        for seed in range(niter):
            A = Star(T=self.T, L=self.L, frac_err=self.frac_err,
                     seed=seed, x=self.x, name=self.name)
            A.fit_bb_Single(use_priors=True)
            if hasattr(A, 'logT_Single'):
                _fit_params_Single = dict(name=A.name,
                                          seed=A.seed,
                                          logT=A.logT,
                                          logL=A.logL,
                                          logT_Single=A.logT_Single,
                                          logL_Single=A.logL_Single)
            else:
                _fit_params_Single = dict(name=A.name,
                                          seed=A.seed,
                                          logT=A.logT,
                                          logL=A.logL,
                                          logT_Single=np.nan,
                                          logL_Single=np.nan)
            fit_params.append(_fit_params_Single)
        df_fit_params = pd.DataFrame(fit_params)
        for column in ['logT', 'logL', 'logT_Single', 'logL_Single']:
            df_fit_params[column[3:]] = 10**df_fit_params[column]
        good_fits = (np.isfinite(df_fit_params.logT_Single))
        self.convergence_rate = np.sum(good_fits)/niter
        df_fit_params['flag_good_fit'] = good_fits
        df_fit_params_good = df_fit_params[good_fits]

        df_summary = pd.Series()
        for column in ['logT', 'logL', 'T', 'L',]:
            df_summary[column] = df_fit_params_good.iloc[0][column]

        for column in ['logT_Single', 'logL_Single', 'T_Single', 'L_Single']:
            q16, q50, q84 = np.nanquantile(
                df_fit_params_good[column], [0.16, 0.50, 0.84])
            df_summary[column+'_16'] = q16
            df_summary[column+'_50'] = q50
            df_summary[column+'_84'] = q84
            df_summary['e_'+column+'_upper'] = q84-q50
            df_summary['e_'+column+'_lower'] = q50-q16

        self.df_error_fit_params = df_fit_params
        self.df_error_summary = df_summary

        if verbose:
            print('%s\n%s' % (self.name, '-'*max(5, len(self.name))))
            print('T_in  = [%f]\nL_in  = [%f]' % (self.T.value, self.L.value))
            print('T_fit = [%f +%f-%f]' % (df_summary['T_Single_50'],
                  df_summary['e_T_Single_upper'], df_summary['e_T_Single_lower']))
            print('L_fit = [%f +%f-%f]' % (df_summary['L_Single_50'],
                  df_summary['e_L_Single_upper'], df_summary['e_L_Single_lower']))
            print('Convergence rate:%.2f' % self.convergence_rate)

    def plot_estimated_errors(self, ax=None, plot_name: str = None) -> None:
        """
        Plot estimated errors on the Hertzsprung-Russell diagram.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axis to plot on. If None, a new figure is created.
        plot_name : str, optional
            Path to save the plot. If None, the plot is not saved. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        Plotter.plot_isochrone_and_wd(ax=ax)
        Plotter.plot_error_estimate_Single(self, ax=ax, plot_name=plot_name)


class Binary:
    @u.quantity_input
    def __init__(self,
                 T_A: u.Kelvin,
                 T_B: u.Kelvin,
                 L_A: u.solLum,
                 L_B: u.solLum,
                 frac_err: Union[float, np.ndarray, list] = 0,
                 seed: int = 0,
                 D: u.pc = 10 * u.pc,
                 threshold_ewr: float = 5,
                 x: np.ndarray = np.linspace(3.1, 4.7, 9),
                 name: str = ''
                 ) -> None:
        """
        Initialize a Binary system with two stars.

        Parameters
        ----------
        T_A : u.Kelvin
            Temperature of the first star.
        T_B : u.Kelvin
            Temperature of the second star.
        L_A : u.solLum
            Luminosity of the first star.
        L_B : u.solLum
            Luminosity of the second star.
        frac_err : float or np.ndarray or list, optional
            Fractional error in flux. Defaults to 0.
        seed : int, optional
            Random seed. Defaults to 0.
        D : u.pc, optional
            Distance to the binary system. Defaults to 10 parsecs.
        threshold_ewr : float, optional
            Detection threshold for EWR to identify poorly fitting filters (with |EWR|>threshold_ewr). Defaults to 5.
        x : np.ndarray, optional
            log(Wavelengths [Angstrom]) for the spectrum/SED. Pivot wavelengths of the filters. Defaults to `np.linspace(3.1, 4.7, 9)`. 
        name : str, optional
            Name of the binary system. Defaults to an empty string.
        """
        self.A = Star(T=T_A,
                      L=L_A,
                      D=D,
                      frac_err=0,
                      threshold_ewr=threshold_ewr,
                      x=x)
        self.B = Star(T=T_B,
                      L=L_B,
                      D=D,
                      frac_err=0,
                      threshold_ewr=threshold_ewr,
                      x=x)

        self.threshold_ewr = threshold_ewr
        self.D = D
        self.x = x
        self.frac_err = frac_err
        self.seed = seed
        self.name = name

        self.spectrum = self.A.spectrum + self.B.spectrum
        self.spectrum.add_noise(self.frac_err, seed=seed)
        self.spectrum_ratio = self.B.spectrum/self.A.spectrum

    def plot(self, ax=None, **kwargs) -> None:
        """
        Plot the input spectrum of the binary system and its individual components.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        **kwargs
            Additional keyword arguments passed to the plotting function.
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

    def fit_bb_Single(self, **kwargs) -> None:
        """
        Fit a single blackbody model to the binary system's spectrum.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the fitting function.
        """
        Fitter.bb_Single(self.spectrum, source=self, **kwargs)

    def fit_bb_Double(self,
                      use_priors: bool = False,
                      threshold_primary_match: None | float = None,
                      **kwargs
                      ) -> None:
        """
        Fit a double blackbody model to the binary system's spectrum.

        Parameters
        ----------
        use_priors : bool, optional
            Whether to use prior values for fitting. Defaults to False.
        threshold_primary_match : None or float, optional
            Threshold for matching primary star parameter in fitting. Defaults to None.
        **kwargs
            Additional keyword arguments passed to the fitting function.
        """
        if use_priors:
            p0 = [self.A.T.value, self.A.logsf,
                  self.B.T.value, self.B.logsf]
        if 'p0' in kwargs:
            p0 = kwargs['p0']
        if ('p0' not in kwargs) & (not use_priors):
            p0 = [5000, -20, 6000, -20]
        Fitter.bb_Double(self.spectrum, source=self, p0=p0,
                         threshold_primary_match=threshold_primary_match, **kwargs)

    def plot_fitted(self, mode: str, axes=None, plot_name: None | str = None) -> None:
        """
        Plot the fitted spectrum of the binary system.

        Parameters
        ----------
        mode : str
            Mode for plotting (e.g., fitting results).
        axes : None or matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new plot is created.
        plot_name : None or str, optional
            Path to save the plot. If None, the plot is not saved.

        Returns
        -------
        None
        """
        Plotter.plot_fitted(self, mode, axes=axes)
        if plot_name is not None:
            if not os.path.exists(os.path.dirname(plot_name)):
                os.makedirs(os.path.dirname(plot_name))
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close()

    def estimate_errors(self,
                        niter: int = 100,
                        verbose: bool = True,
                        threshold_primary_match: float = 0.10
                        ) -> None:
        """
        Estimate errors by fitting models multiple times and evaluating the parameter spread.

        Parameters
        ----------
        niter : int, optional
            Number of random iterations for error estimation. Defaults to 100.
        verbose : bool, optional
            If True, print results and convergence information. Defaults to True.
        threshold_primary_match : float, optional
            Threshold for matching primary star parameters in fitting. Defaults to 0.10.
        """
        fit_params = []
        for seed in range(niter):
            AB = Binary(T_A=self.A.T, L_A=self.A.L, T_B=self.B.T, L_B=self.B.L,
                        frac_err=self.frac_err, seed=seed, x=self.x, name=self.name)
            AB.fit_bb_Double(
                use_priors=True, threshold_primary_match=threshold_primary_match)
            fit_params_Single = dict(name=AB.name,
                                     seed=AB.seed,
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
        for column in ['logT_A', 'logL_A', 'logT_B', 'logL_B',
                       'logT_A_Double', 'logL_A_Double', 'logT_B_Double', 'logL_B_Double']:
            df_fit_params[column[3:]] = 10**df_fit_params[column]

        good_fits = (np.isfinite(df_fit_params.logT_A_Double))
        self.convergence_rate = np.sum(good_fits)/niter
        df_fit_params['flag_good_fit'] = good_fits
        df_fit_params_good = df_fit_params[good_fits]

        df_summary = pd.Series()
        for column in ['logT_A', 'logL_A', 'logT_B', 'logL_B',
                       'T_A', 'L_A', 'T_B', 'L_B']:
            df_summary[column] = df_fit_params_good.iloc[0][column]

        for column in ['logT_A_Double', 'logL_A_Double', 'logT_B_Double', 'logL_B_Double',
                       'T_A_Double', 'L_A_Double', 'T_B_Double', 'L_B_Double',]:
            q16, q50, q84 = np.nanquantile(
                df_fit_params_good[column], [0.16, 0.50, 0.84])
            df_summary[column+'_16'] = q16
            df_summary[column+'_50'] = q50
            df_summary[column+'_84'] = q84
            df_summary['e_'+column+'_upper'] = q84-q50
            df_summary['e_'+column+'_lower'] = q50-q16

        self.df_error_fit_params = df_fit_params
        self.df_error_summary = df_summary
        if verbose:
            print('%s\n%s' % (self.name, '-'*max(5, len(self.name))))
            print('T_in  = [%f]\t [%f]\nL_in  = [%f]\t [%f]' %
                  (self.A.T.value, self.B.T.value, self.A.L.value, self.B.L.value))
            print('T_fit = [%f +%f-%f]\t[%f +%f-%f]' % (df_summary['T_A_Double_50'], df_summary['e_T_A_Double_upper'], df_summary['e_T_A_Double_lower'],
                                                        df_summary['T_B_Double_50'], df_summary['e_T_B_Double_upper'], df_summary['e_T_B_Double_lower']))
            print('L_fit = [%f +%f-%f]\t[%f +%f-%f]' % (df_summary['L_A_Double_50'], df_summary['e_L_A_Double_upper'], df_summary['e_L_A_Double_lower'],
                                                        df_summary['L_B_Double_50'], df_summary['e_L_B_Double_upper'], df_summary['e_L_B_Double_lower']))
            print('Convergence rate:%.2f' % self.convergence_rate)

    def plot_estimated_errors(self, ax=None, plot_name: None | str = None) -> None:
        """
        Plot estimated errors and median values on the HR diagram.

        Parameters
        ----------
        ax : None or matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        plot_name : None or str, optional
            Path to save the plot. If None, the plot is not saved.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        Plotter.plot_isochrone_and_wd(ax=ax)
        Plotter.plot_error_estimate_Double(self, ax=ax, plot_name=plot_name)

    def evaluate_pseudo_secondaries(self, grid_size: int = 5, niter: int = 100, refit: bool = False) -> None:
        """
        Evaluate a grid of pseudo-secondaries near the HRD position of B component.

        Parameters
        ----------
        grid_size : int, optional
            Size of the grid for parameter exploration. grid_size*grid_size pseudo-secondaries wioll be created. Defaults to 5.
        niter : int, optional
            Number of random iterations for each parameter set. Defaults to 100.
        refit : bool, optional
            Whether to refit the parameters during the evaluation. Defaults to False.
        """
        df_fit_params_good = self.df_error_fit_params[self.df_error_fit_params['flag_good_fit']]
        _logT_min, logT_max = np.quantile(
            df_fit_params_good['logT_B_Double'], [0.05, 0.95])
        _logL_min, logL_max = np.quantile(
            df_fit_params_good['logL_B_Double'], [0.05, 0.95])
        logT_B_list = np.linspace(_logT_min-0.1, logT_max+0.1, grid_size)
        logL_B_list = np.linspace(_logL_min-0.5, logL_max+0.5, grid_size)
        logT_B_list, logL_B_list = np.meshgrid(logT_B_list, logL_B_list)

        self.grid = Grid(T_A=self.A.T,
                         L_A=self.A.L,
                         niter=niter,
                         frac_err=self.frac_err,
                         name=self.name,
                         logT_B_list=logT_B_list.flatten(),
                         logL_B_list=logL_B_list.flatten(),
                         x=self.x)
        self.grid.calculate_params(refit=refit)

    def plot_pseudo_secondaries(self, ax=None, plot_name: None | str = None) -> None:
        """
        Plot the grid of pseudo-secondaries.

        Parameters
        ----------
        ax : None or matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        plot_name : None or str, optional
            Path to save the plot. If None, the plot is not saved.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        Plotter.plot_pseudo_secondaries_Double(
            self, ax=ax, plot_name=plot_name)
        Plotter.plot_isochrone_and_wd(ax=ax)

    def plot_error_and_pseudo_secondaries(self, ax=None, plot_name: None | str = None) -> None:
        """
        Plot both the error estimation and the grid of pseudo-secondaries.

        Parameters
        ----------
        ax : None or matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        plot_name : None or str, optional
            Path to save the plot. If None, the plot is not saved.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        self.plot_pseudo_secondaries(ax=ax)
        self.plot_estimated_errors(ax=ax)
        ax.invert_xaxis()
        Plotter.savefig(plot_name)


class Grid:
    def __init__(self,
                 T_A: u.Kelvin,
                 L_A: u.solLum,
                 niter: int = 1,
                 frac_err: Union[float, np.ndarray, list] = 0.001,
                 D: u.pc = 10 * u.pc,
                 threshold_nfilters: int = 3,
                 threshold_ewr: float = 5,
                 threshold_primary_match: float = 0.1,
                 threshold_convergence_rate: float = 0.5,
                 x: np.ndarray = np.linspace(3.1, 4.7, 9),
                 name: str = '',
                 logT_B_list: list[float] = None,
                 logL_B_list: list[float] = None
                 ) -> None:
        """
        Initializes the Grid of secondaries with the given parameters.

        Parameters
        ----------
        T_A : u.Kelvin
            Temperature of star A.
        L_A : u.solLum
            Luminosity of star A.
        niter : int, optional
            Number of random iterations for fitting, by default 1.
        frac_err : float or np.ndarray or list, optional
            Fractional error in flux. Must be greater than 0, by default 0.001.
        D : u.pc, optional
            Distance to the binary system, by default 10 pc.
        threshold_nfilters : int, optional
            Threshold for the detection of poorly fitting filters, by default 3.
        threshold_ewr : float, optional
            Detection threshold for EWR to identify poorly fitting filters, by default 5.
        threshold_primary_match : float
            Acceptable fractional error in the primary parameters (T and sf).
        threshold_convergence_rate : float
            Rate threshold for assessing the convergence, by default 0.5.
        x : np.ndarray, optional
            Log(wavelength [Angstrom]) range for spectrum/SED. Pivot wavelengths of fitlters. By default `np.linspace(3.1, 4.7, 9)`.
        name : str, optional
            Name for the grid, by default an empty string.
        logT_B_list : list of float, optional
            List of log(temperatures [K]) for star B.
        logL_B_list : list of float, optional
            List of log(luminosities [solLum]) for star B.

        Raises
        ------
        NotImplementedError
            If `frac_err` is set to 0.
        ValueError
            If `logT_B_list` and `logL_B_list` have different lengths.
        """
        if (frac_err is 0) | (frac_err is 0.0):
            raise NotImplemented(
                'frac_err=0 is trivial and not implemented. Provide frac_err>0.')

        self.T_A = T_A
        self.L_A = L_A
        self.frac_err = frac_err
        self.D = D
        self.niter = niter
        self.threshold_nfilters = threshold_nfilters
        self.threshold_ewr = threshold_ewr
        self.threshold_primary_match = threshold_primary_match
        self.threshold_convergence_rate = threshold_convergence_rate
        self.x = x
        self.name = name

        self.logT_A = np.log10(T_A.value)
        self.logL_A = np.log10(L_A.value)

        if len(logT_B_list) != len(logL_B_list):
            raise ValueError(
                'logT_B_list and logL_B_list should have same length.')
        self.n_B = len(logT_B_list)
        self.logT_B_list, self.logL_B_list = logT_B_list, logL_B_list

    def get_AB_fit(self,
                   T_A: u.Kelvin,
                   T_B: u.Kelvin,
                   L_A: u.solLum,
                   L_B: u.solLum,
                   frac_err: Union[float, np.ndarray],
                   seed: int,
                   name: str
                   ) -> 'Binary':
        """
        A helper function that creates a `Binary` instance and performs a fit.

        Parameters
        ----------
        T_A : u.Kelvin
            Temperature of star A.
        T_B : u.Kelvin
            Temperature of star B.
        L_A : u.solLum
            Luminosity of star A.
        L_B : u.solLum
            Luminosity of star B.
        frac_err : float or np.array
            Fractional error in flux.
        seed : int
            Random seed for fitting.
        name : str
            Name for the binary fit.

        Returns
        -------
        Binary
            A `Binary` instance with fitted parameters.
        """
        AB = Binary(T_A, T_B, L_A, L_B, frac_err=frac_err,
                    seed=seed, name=name,
                    D=self.D, x=self.x,
                    threshold_ewr=self.threshold_ewr)

        AB.fit_bb_Single(p0=[AB.A.T.value, AB.A.logsf])

        if hasattr(AB, 'misfits_ewr_Single'):
            if AB.misfits_ewr_Single >= self.threshold_nfilters:
                AB.fit_bb_Double(
                    use_priors=True, threshold_primary_match=self.threshold_primary_match)
        return AB

    def get_sed_params(self, AB: 'Binary') -> pd.DataFrame:
        """
        A helper function that extracts SED parameters from the fitted `Binary` instance.

        Parameters
        ----------
        AB : Binary
            The `Binary` instance to extract parameters from.

        Returns
        -------
        pd.DataFrame
            DataFrame containing SED parameters.
        """
        df_sed = pd.DataFrame()
        df_sed['x'] = AB.x
        df_sed['seed'] = AB.seed
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

    def get_AB_fit_params(self, AB: 'Binary') -> dict[str, float]:
        """
        A helper function that retrieves fitting parameters from the `Binary` instance.

        Parameters
        ----------
        AB : Binary
            The `Binary` instance to extract fit parameters from.

        Returns
        -------
        dict of str to float
            Dictionary containing fit parameters.
        """
        fit_params_priors = dict(name=AB.name,
                                 seed=AB.seed,
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

    def process_fit_params(self, df_fit_params: pd.DataFrame) -> None:
        """
        Helper function that processes the fitting parameters to calculate medians and errors.

        Parameters
        ----------
        df_fit_params : pd.DataFrame
            DataFrame of fit parameters.
        """
        df_fit_params['id'] = df_fit_params['name'].str.extract(
            '(\d+)').astype(int)
        for column in ['logT_A', 'logL_A', 'logT_B', 'logL_B',
                       'logT_Single', 'logL_Single',
                       'logT_A_Double', 'logL_A_Double', 'logT_B_Double', 'logL_B_Double']:
            df_fit_params[column[3:]] = 10**df_fit_params[column]
        drop_column_list = ['name', 'seed']
        _group = df_fit_params.drop(columns=drop_column_list).groupby(by='id')
        q50 = _group.median()

        drop_column_list = ['name', 'seed',
                            'logT_A', 'logL_A', 'logT_B', 'logL_B',
                            'T_A', 'L_A', 'T_B', 'L_B']
        _group = df_fit_params.drop(columns=drop_column_list).groupby(by='id')
        q16 = _group.quantile(0.16)
        q84 = _group.quantile(0.84)

        column_list = ['logT_Single', 'logL_Single',
                       'logT_A_Double', 'logL_A_Double', 'logT_B_Double', 'logL_B_Double',
                       'T_Single', 'L_Single',
                       'T_A_Double', 'L_A_Double', 'T_B_Double', 'L_B_Double']

        for column in column_list:
            q50.rename(columns={column: column+'_50'}, inplace=True)
            q16.rename(columns={column: column+'_16'}, inplace=True)
            q84.rename(columns={column: column+'_84'}, inplace=True)
        df_fit_params_summary = pd.concat([q50, q16, q84], axis=1)

        for column in column_list:
            df_fit_params_summary['e_'+column+'_upper'] = df_fit_params_summary[column+'_84'] -\
                df_fit_params_summary[column+'_50']
            df_fit_params_summary['e_'+column+'_lower'] = df_fit_params_summary[column+'_50'] -\
                df_fit_params_summary[column+'_16']

        df_fit_params_summary['convergence_rate'] = _group.count()[
            'logT_A_Double']/self.niter
        df_fit_params_summary['frac_e_T_A'] = (df_fit_params_summary['e_T_A_Double_lower'] +
                                               df_fit_params_summary['e_T_A_Double_upper'])/2/df_fit_params_summary['T_A_Double_50']
        self.df_fit_params = df_fit_params
        self.df_fit_params_summary = df_fit_params_summary.reset_index()

    def calculate_params(self, refit: bool = False) -> None:
        """
        Calculates and saves fitting parameters and SEDs.

        Parameters
        ----------
        refit : bool, optional
            If True, refits and recalculates parameters, by default False.
        """
        if not os.path.exists('outputs/%s' % self.name):
            os.makedirs('outputs/%s' % self.name)
        fit_params_name = 'outputs/%s/fit_params.csv' % self.name
        fit_params_summary_name = 'outputs/%s/fit_params_summary.csv' % self.name
        sed_niter0_name = 'outputs/%s/sed_niter0.csv' % self.name

        # If the files exists, they will be read (depends on refit = True or False)
        if os.path.isfile(fit_params_name):
            if not refit:
                self.df_fit_params = pd.read_csv(fit_params_name)
                self.df_fit_params_summary = pd.read_csv(
                    fit_params_summary_name)
                self.df_sed_niter0 = pd.read_csv(sed_niter0_name)
                return

        df_sed_niter0 = pd.DataFrame()
        fit_params = []

        for idx in tqdm(range(self.n_B), desc='%s: Grid params ' % self.name, leave=False):
            # for idx in (range(self.n_B)):
            _T = 10**self.logT_B_list[idx] * u.K
            _L = 10**self.logL_B_list[idx] * u.solLum
            for seed in range(self.niter):
                name = 'id%d_niter%d_' % (idx, seed)
                AB = self.get_AB_fit(self.T_A, _T, self.L_A, _L,
                                     self.frac_err, seed, name)
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
        self.process_fit_params(pd.DataFrame(fit_params))

        self.df_fit_params.to_csv(fit_params_name, index=False)
        self.df_fit_params_summary.to_csv(fit_params_summary_name, index=False)
        self.df_sed_niter0.to_csv(sed_niter0_name, index=False)

    def plot_skeleton(self, ax=None, zorder: int = 0) -> None:
        """
        Plots the skeleton of the data on the provided axis.

        Parameters
        ----------
        ax : plt.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        zorder : int, optional
            The z-order for the scatter plot, by default 0.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.logT_A, self.logL_A, s=50, marker='s', zorder=zorder,
                   facecolors='none', edgecolors='k')
        ax.scatter(self.logT_B_list, self.logL_B_list, c='0.3',
                   zorder=-1, s=10, marker='.', rasterized=True)

    def plot_sed_patches(self, param: str = 'ewr_Single', ax=None) -> None:
        """
        Plots the SED patches on the provided axis. The '|' markers are coloured according to `param` variable.

        Parameters
        ----------
        param : str
            The parameter to color the SED patches. Defaults to 'ewr_Single'
        ax : plt.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        """
        def x_to_logT(x: np.ndarray, logT: float, delta_logT: float, x_new: np.ndarray, buffer: int = 4) -> float:
            """
            Perform a linear interpolation to estimate logT values based on a range of x values.

            Parameters
            ----------
            x : array-like
                The original x values (independent variable).
            logT : float
                The base logT value.
            delta_logT : float
                The range of logT values.
            x_new : np.ndarray
                The new x value for which logT needs to be estimated.
            buffer : int, optional
                A buffer to adjust spacing. Defaults to 4.

            Returns
            -------
            float
                The estimated logT value for the given x_new.
            """
            x1, y1 = x.min(), logT + delta_logT/buffer
            x2, y2 = x.max(), logT - delta_logT/buffer
            logT_new = y1 + (y2-y1) * (x_new-x1) / (x2-x1)
            return logT_new

        def y_to_logL(y: np.ndarray, ymin: float, yptp: float, height: float = 1.2, offset: float = 0.2) -> np.ndarray:
            """
            Convert y values to logL values based on a given height and offset.

            Parameters
            ----------
            y : array-like
                The y values to be converted.
            ymin : float
                The minimum y value.
            yptp : float
                The peak-to-peak range of y values.
            height : float, optional
                The scaling factor for the y values. Defaults to 1.2.
            offset : float, optional
                The offset to adjust the y values. Defaults to 0.2.

            Returns
            -------
            array-like
                The converted logL values.
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
                                vmin=-self.threshold_ewr*5./3., vmax=self.threshold_ewr*5./3.,
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

    def plot_Double_fitting_lines(self,
                                  niter: int = None,
                                  ax=None,
                                  plot_poor: bool = True
                                  ) -> None:
        """
        Plots the double fitting lines on the provided axis.

        Parameters
        ----------
        niter : int, optional
            The iteration number for the fitting. If None, median values are used.
        ax : plt.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        plot_poor : bool, optional
            Whether to plot poorly fitting results (with convergence_rate<0.5), by default True.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if niter is None:
            df = self.df_fit_params_summary
            suffix = '_50'
        else:
            df = self.df_fit_params
            df = df[df['name'].str.contains('_niter%d_' % niter)].reset_index()
            suffix = ''

        logT_B_list = df.logT_B.unique()
        logL_B_list = df.logL_B.unique()
        color_list = plt.cm.twilight_r(np.linspace(
            0, 1, len(logT_B_list)+4))[2:-2]  # creates array of N colors

        check_1 = (self.df_fit_params_summary.convergence_rate >=
                   self.threshold_convergence_rate)
        check_2 = (self.df_fit_params_summary.frac_e_T_A <
                   self.threshold_primary_match)
        check_3 = (np.isfinite(df['logT_A_Double'+suffix]))
        good_fit = (check_1 & check_2)
        df_good = df[check_3 & good_fit].reset_index()
        df_poor = df[check_3 & ~good_fit].reset_index()

        ax.plot([df.logT_A, df['logT_A_Double'+suffix]], [df.logL_A,
                df['logL_A_Double'+suffix]], c='k', marker='.', ms=0.5)
        for idx, logT_B in enumerate(logT_B_list):
            for jdx, logL_B in enumerate(logL_B_list):
                _df = df_good[(df_good.logT_B == logT_B) &
                              (df_good.logL_B == logL_B)]
                ax.plot([_df.logT_B, _df['logT_B_Double'+suffix]],
                        [_df.logL_B, _df['logL_B_Double'+suffix]],
                        c=color_list[idx], ls='-',
                        marker='.', ms=1)
                if plot_poor:
                    _df = df_poor[(df_poor.logT_B == logT_B) &
                                  (df_poor.logL_B == logL_B)]
                    ax.plot([_df.logT_B, _df['logT_B_Double'+suffix]],
                            [_df.logL_B, _df['logL_B_Double'+suffix]],
                            c=color_list[idx], ls='--',
                            marker='.', ms=1)

    def plot_Double_fitting_points(self, ax=None, noisy: bool = False) -> None:
        """
        Plot the double fitting points on the provided axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        noisy : bool, optional
            If True, plot noisy data points with varying sizes. Defaults to False.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        df = self.df_fit_params_summary
        logT_B_list = np.sort(np.unique(df.logT_B))
        check_1 = (df.convergence_rate >= self.threshold_convergence_rate)
        check_2 = (df.frac_e_T_A < self.threshold_primary_match)
        check_3 = (np.isfinite(df.logT_A_Double_50))
        _filter = (check_1 & check_2 & check_3)
        df = df[_filter].reset_index()
        color_list = plt.cm.twilight_r(np.linspace(
            0, 1, len(logT_B_list)+4))[2:-2]  # creates array of N colors

        if noisy:
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                'new_cmap', color_list, N=len(logT_B_list))

            df_noisy = self.df_fit_params
            df_noisy = df_noisy[np.isfinite(
                df_noisy.logT_A_Double)].reset_index()
            s_list = np.linspace(10, 100, len(logT_B_list))
            s_list = (df.logL_B - np.nanmin(df.logL_B))**2
            s_list = s_list/np.nanmax(s_list) * 90 + 10
            s_list = (df_noisy.logL_B - np.nanmin(df_noisy.logL_B))**2
            s_list = s_list/np.nanmax(s_list) * 90 + 10
            p0 = ax.scatter(df_noisy.logT_B_Double, df_noisy.logL_B_Double,
                            c=df_noisy.logT_B, cmap=cmap, zorder=4, s=s_list,
                            marker='$◯$', lw=0.1, alpha=0.5)

        for idx, logTe_B in enumerate(logT_B_list):
            _df = df[(df.logT_B == logTe_B)]
            ax.plot([_df.logT_B, _df.logT_B_Double_50],
                    [_df.logL_B, _df.logL_B_Double_50],
                    c=color_list[idx], alpha=1, ls=':',
                    zorder=0)

            x, y = _df.logT_B_Double_50, _df.logL_B_Double_50
            e_x_lower, e_x_upper = _df.e_logT_B_Double_lower, _df.e_logT_B_Double_upper
            e_y_lower, e_y_upper = _df.e_logL_B_Double_lower, _df.e_logL_B_Double_upper
            e_x = [e_x_lower, e_x_upper]
            e_y = [e_y_lower, e_y_upper]
            ax.errorbar(x, y, xerr=e_x, yerr=e_y,
                        ls='', ecolor=color_list[idx], marker='d', color=color_list[idx],
                        lw=1)

    def plot_Double_fitting_T2err(self,
                                  ax=None,
                                  cax=None,
                                  colorbar: bool = True,
                                  **kwargs
                                  ) -> None:
        """
        Plot the percentage T_2 errors for the fitted secondary grid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        cax : matplotlib.axes.Axes, optional
            The colorbar axes to use. If None, a new colorbar is created.
        colorbar : bool, optional
            If True, include a colorbar. Defaults to True.
        **kwargs : dict
            Additional keyword arguments passed to `ax.scatter`.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        df = self.df_fit_params_summary

        check_1 = (df.convergence_rate >= self.threshold_convergence_rate)
        check_2 = (df.frac_e_T_A < self.threshold_primary_match)
        check_3 = (np.isfinite(df.logT_A_Double_50))
        _filter = (check_1 & check_2 & check_3)

        df = df[_filter].reset_index()

        cmap = plt.get_cmap('cool', 3)   # 3 discrete colors
        c = (df.e_T_B_Double_lower+df.e_T_B_Double_upper) / \
            2/df.T_B_Double_50 * 100
        p0 = ax.scatter(df.logT_B, df.logL_B, marker='d',
                        c=c, vmin=0, vmax=15, cmap=cmap, rasterized=True, **kwargs)
        if colorbar:
            plt.colorbar(p0, extend='max', cax=cax,
                         label='T$_{2,\,err}$ (%)')  # ε$_T$

    def plot(self, plot_name: str = None) -> None:
        """
        Generate a series of plots and save to a file if specified.

        Parameters
        ----------
        plot_name : str, optional
            The path to save the plot. If None, the plot is not saved.
        """
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 13))
        fig.subplots_adjust(hspace=0.3)
        self.plot_skeleton(ax=ax[0, 0])
        self.plot_skeleton(ax=ax[0, 1])
        self.plot_skeleton(ax=ax[1, 0])
        self.plot_skeleton(ax=ax[1, 1])

        for axi in [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]:
            Plotter.plot_isochrone_and_wd(axi)
        self.plot_sed_patches(ax=ax[0, 0], param='ewr_Single')
        self.plot_Double_fitting_lines(ax=ax[0, 1], niter=0)
        self.plot_Double_fitting_lines(ax=ax[1, 0])
        self.plot_Double_fitting_points(ax=ax[1, 1], noisy=False)
        ax[0, 0].set_title('SEDs of secondaries coloured with EWR\nOnly 1st random realisation')
        ax[0, 1].set_title('Input and recovered positions connected by lines\nOnly 1st random realisation')
        ax[1, 0].set_title('Input and recovered positions connected by lines\nAverage of all random realisation')
        ax[1, 1].set_title('Input and recovered positions connected by lines\nDiamond markers show Monte-Carlo errors\nAverage of all random realisation')
        Plotter.savefig(plot_name)
