import pytest
import re
from astropy import units as u
import numpy as np
import sed_analysis_tools as st

def test_list_pivot_wavelength():
    filter_set = st.FilterSet(
        list_pivot_wavelengths=(100, 200, 300) * u.Angstrom)
    assert ((filter_set.list_pivot_wavelengths ==
            [100, 200, 300]*u.Angstrom).all())


def test_list_filter_names():
    filter_set = st.FilterSet(
        list_filter_names=["2MASS/2MASS.J", "2MASS/2MASS.H"])
    assert (np.allclose(filter_set.list_pivot_wavelengths,
            [12358.089, 16457.504]*u.Angstrom))


def test_incorrect_filter_name():
    list_filter_names = ["2MASS2MASS.U"]
    with pytest.raises(ValueError, match=re.escape("Filter ID (2MASS2MASS.U) not found in SVO Filter Profile Service.")):
        st.FilterSet(list_filter_names=list_filter_names)


def test_Star_error_estimation():
    starA = st.Star(T=8000*u.K,
                    L=1*u.solLum,
                    frac_err=0.01,
                    seed=0,
                    D=10*u.pc,
                    threshold_ewr=5.0,
                    name='A')
    starA.estimate_errors(verbose=False)
    assert (np.allclose(starA.df_error_summary[[
            'T_Single_50', 'L_Single_50']].values, [8000, 1], rtol=0.01))


def test_failed_Star_fit():
    starA = st.Star(T=800*u.K,
                    L=1*u.solLum,
                    frac_err=0.3,
                    seed=0,
                    D=10*u.pc,
                    threshold_ewr=5.0,
                    name='A')
    with pytest.warns(UserWarning, match=re.escape("A: Fit rejected due to being close to the parameter (T, logsf) boundary.")):
        starA.fit_bb_Single()
    assert ~(hasattr(starA, 'T_Single'))


def test_Binary_fit():
    binary = st.Binary(
        T_A=6000 * u.K,
        T_B=9000 * u.K,
        L_A=1 * u.solLum,
        L_B=0.5 * u.solLum,
        frac_err=0.01,
        seed=2,
        D=10 * u.pc,
        name="AB",
    )

    binary.fit_bb_Double(use_priors=True)
    assert (np.allclose([binary.T_A.value, binary.L_A.value,
            binary.T_B.value, binary.L_B.value], [6000, 1, 9000, 0.5], rtol=0.1))


def test_Binary_error_estimation():
    binary = st.Binary(
        T_A=6000 * u.K,
        T_B=9000 * u.K,
        L_A=1 * u.solLum,
        L_B=0.5 * u.solLum,
        frac_err=0.01,
        seed=2,
        D=10 * u.pc,
        name="AB",
    )
    binary.estimate_errors()
    assert (np.allclose(binary.df_error_summary[[
            'T_A_Double_50', 'L_A_Double_50']].values, [6000, 1], rtol=0.01))


def test_primary_threshold_warning():
    binary = st.Binary(
        T_A=6000 * u.K,
        T_B=20000 * u.K,
        L_A=1 * u.solLum,
        L_B=1 * u.solLum,
        frac_err=0.3,
        seed=2,
        name="AB",
    )
    with pytest.warns(UserWarning, match="AB: Fit rejected due to poor primary fit."):
        binary.fit_bb_Double(use_priors=True, threshold_primary_match=0.01)

def test_Binary_pseudo_secondaries():
    binary = st.Binary(
        T_A=6000 * u.K,
        T_B=15000 * u.K,
        L_A=1 * u.solLum,
        L_B=1 * u.solLum,
        frac_err=0.01,
        seed=2,
        name="AB",
    )
    binary.evaluate_pseudo_secondaries(niter=10, grid_size=3)
    assert(np.allclose(binary.df_error_summary[['T_A_Double_50','L_A_Double_50']], [6000,1], rtol=0.01))

def test_bad_Binary_pseudo_secondaries():
    import warnings
    with warnings.catch_warnings():
        # ignore all warnings
        warnings.filterwarnings('ignore')
        binary = st.Binary(
            T_A=6000 * u.K,
            T_B=9000 * u.K,
            L_A=1 * u.solLum,
            L_B=1 * u.solLum,
            frac_err=0.3,
            seed=2,
            name="AB",
        )
        binary.evaluate_pseudo_secondaries(niter=10, grid_size=3)
        assert(binary.grid.df_fit_params_summary['convergence_rate'][0] == 0.3)