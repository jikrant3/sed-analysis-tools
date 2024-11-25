from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))


VERSION = '1.2.0'
DESCRIPTION = 'Analysing SED fitting of single and binary stars'
LONG_DESCRIPTION = 'A package that allows to model stars and binaries as blackbody source, and analyse the recoverability of the stellar parameters for a given filter set and noise characteristics.'

# Setting up
setup(
    name="sed_analysis_tools",
    version=VERSION,
    author="Vikrant V. Jadhav",
    author_email="<vjadhav@uni-bonn.de>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['astropy', 'matplotlib', 'numpy', 'pandas>=2.0.0', 'scipy', 'tqdm'],
    keywords=['python', 'black body', 'spectral energy distribution', 'binary system'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Astronomy"
    ]
)
