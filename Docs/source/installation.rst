Installing ANTARESS
===================

Latest stable release
---------------------

``ANTARESS`` will be available on the PyPI repository in the future. In the meantime, you can retrieve the code and install it following the instructions below.


Current version
---------------

``ANTARESS`` is hosted at https://gitlab.unige.ch/bourrier/antaress.
To download the repository containing the current version, you can use::

    git clone https://gitlab.unige.ch/bourrier/antaress.git

This version is being developed and may not be stable.

Manual installation
-------------------

You need to install the following packages to run ``ANTARESS``:

- Standard packages
    Install `scipy <https://scipy.org/>`_, `lmfit <https://lmfit.github.io/lmfit-py/>`_, batman-package, astropy, emcee, pathos, pandas, dace_query, statsmodels, PyAstronomy using::
    
        pip install package         

- Resampling package 
    - `bindensity documentation <https://obswww.unige.ch/~delisle/bindensity/doc/>`_
    - install as a standard package::
    
        pip install bindensity

    - do not use ``bindensity`` with non-continuous grids, as it will mess up with banded covariance matrixes.
    - beware when masking spectral ranges with ``ANTARESS``, as undefined pixels (set to nan values) are propagated by ``bindensity`` when resampling and will `spread` throughout the workflow.

- Package ``pySME`` 
    - follow these instructions to install ``PySME`` on M1/M2 Macs.
        - install rosetta by running::
        
            softwareupdate --install-rosetta

        - install Homebrew under rosetta by running::

            $ arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

          This installs Homebrew under `/usr/local/bin/brew` instead of the default for arm64 `/opt/homebrew/bin/brew`

        - install `gcc@9  <https://tenderlovemaking.com/2022/01/07/homebrew-rosetta-and-ruby.html>`_ by running::

            $ arch -x86_64 /usr/local/bin/brew install gcc@9

        - create a Conda environment to run under the `intel x64_86 architecture <https://abpcomputing.web.cern.ch/guides/apple_silicon/>`_ and install ``PySME`` in this environment::   

            CONDA_SUBDIR=osx-64 conda create -n envname python=3.11

          then::

            pip install pysme-astro

    - follow these instructions to install ``PySME`` on older Macs
        - install gcc9 by running::
            
            brew install gcc@9

        - then ``PySME`` by running::
        
            pip install pysme-astro

- Package ``KitCat``
    - install `gsl <https://www.gnu.org/software/gsl/>`_ by running::
        
        brew install gsl

    - set up the path to your local python installation in the `setup_lbl_fit.py <https://gitlab.unige.ch/bourrier/antaress/-/tree/0d7232f1a1b39757beb8a52762b9e95fd33b2591/Method/ANTARESS_conversions/KitCat/setup_lbl_fit.py>`_ and run::
    
        python setup_lbl_fit.py build
        
      Then copy the compiled file ``calculate_RV_line_by_line3.cpython-XX-darwin.so`` into your `KitCat/ <https://gitlab.unige.ch/bourrier/antaress/-/tree/0d7232f1a1b39757beb8a52762b9e95fd33b2591/Method/ANTARESS_conversions/KitCat/>`_ directory.  
