# sed-analysis-tools
A python module to model, fit and analyse single and binary spectral energy distributions (SEDs). This theory behind the code, and the method it implements, are described in Jadhav (2024, in prep).

## Documentation at [jikrant3.github.io/sed-analysis-tools](https://jikrant3.github.io/sed-analysis-tools)

*Requirements*: `python`>=3.12.2, `astropy`>=6.0.1, `matplotlib`>=3.8.4, `numpy`>=1.26.4, `pandas`>=2.2.1, `scipy`>=1.13.0, `tqdm`>=4.57.0

## Installation
- Add the `src/sed_analysis_tools.py` to your working directory.
   - ⚠️ Edit the `DIR_MODELS` in the `sed_analysis_tools.py` file according to where your model files are located

- Download the required models from [models_and_tools](https://github.com/jikrant3/models_and_tools/tree/main/models).

   - Isochrones: `master_isochrone.csv` ([Parsec isochrones](http://stev.oapd.inaf.it/cmd))

   - WD cooling curves: `master_Bergeron_WD.csv` ([Bergeron models](https://www.astro.umontreal.ca/~bergeron/CoolingModels/))

### `src.sed_analysis_tools.py`
The python module with the code to create and fit single/binary SEDs. Also contails other helper functions which are usefult for estimating errors in SED fiting and other astronomical conversions.

### `manuscript_code.ipynb`
Generates the models required to replicate the contents of the manuscript.

### `manuscript_plots.ipynb`
The jupyter notebook for creating all the plots in the manuscript.

## Citation
If you use this code for your work, kindly include a citation for this code [![DOI](https://zenodo.org/badge/856901189.svg)](https://zenodo.org/doi/10.5281/zenodo.13789847) and accompaning paper.

Jadhav (2024, in prep) _On the detectability and parameterisation of binary stars through spectral energy distributions_.
