Installing ANTARESS
=====================

Latest stable release
-----------------------

``ANTARESS`` will be available on the PyPI repository in the future. In the meantime, you can retrieve the code and install it following the instructions below.


Current version
-----------------------

``ANTARESS`` is hosted at https://gitlab.unige.ch/bourrier/antaress.
To download the repository containing the current version, you can use::

    git clone https://gitlab.unige.ch/bourrier/antaress.git

This version is being developed and may not be stable.

Manual installation
------------

You need to install the following packages to run ``ANTARESS``:

- [ ] run `pip install` with [scipy](https://scipy.org/), lmfit, batman-package, astropy, emcee, pathos, pandas, dace_query, statsmodels, PyAstronomy        
- [ ] resampling package 
- [bindensity documentation](https://obswww.unige.ch/~delisle/staging/bindensity/doc/)
- run `pip install --extra-index-url https://vincent:cestpasfaux@obswww.unige.ch/~delisle/staging bindensity --upgrade`
- do not use routines with non-continuous tables, as it will mess up with banded covariance matrixes
- beware of masking ranges, as undefined pixels (set to nan values) are propagated when resampling or combining profiles in the various pipeline modules.
- [ ] pySME
- follow these instructions to install PySME on M1/M2 Macs.
  - install rosetta by running `softwareupdate --install-rosetta`
  - install Homebrew under rosetta by running  
`$ arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"` 
This installs Homebrew under  
`/usr/local/bin/brew`   
instead of the default for arm64  
`/opt/homebrew/bin/brew`
  - install gcc@9 (see [documentation](https://tenderlovemaking.com/2022/01/07/homebrew-rosetta-and-ruby.html )) by running
`$ arch -x86_64 /usr/local/bin/brew install gcc@9`
  - create a Conda environment to run under the intel x64_86 architecture (see [documentation](https://abpcomputing.web.cern.ch/guides/apple_silicon/)) and install PySME in this environment   
`CONDA_SUBDIR=osx-64 conda create -n envname python=3.11`
then 
`pip install pysme-astro`
- follow these instructions to install PySME on older Macs
  - install gcc9 with `brew install gcc@9`
  - run `pip install pysme-astro`

- [ ] KitCat
- install [gsl](https://www.gnu.org/software/gsl/) with `brew install gsl`
- run `python setup_lbl_fit.py build` after setting up the path to your local python installation in the [setup_lbl_fit.py](ANTARESS_masks/KitCat/setup_lbl_fit.py). Then copy the compiled file `calculate_RV_line_by_line3.cpython-XX-darwin.so` into your ANTARESS/KitCat directory  


