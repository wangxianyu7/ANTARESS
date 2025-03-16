.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

Installing ANTARESS
===================

Latest stable release
---------------------

``ANTARESS`` is available on the PyPI repository. You can install the latest version available with::

    pip install antaress         

Current version
---------------

``ANTARESS`` is hosted on `gitlab <https://gitlab.unige.ch/spice_dune/antaress>`_.
To download the repository containing the current version, you can use::

    git clone https://gitlab.unige.ch/spice_dune/antaress.git

This version is being developed and may not be stable.

The pipeline runs on Mac, Linux, and Windows (``pySME`` is however not available on the latter).

  

Manual installation
-------------------

If you intend to develop or get the latest (unreleased) developments, get the current version as described above, move in the downloaded directory, and install ``ANTARESS`` with::

    pip install -e .

If you want to test updates to the documentation as well, you will need to install the packages `sphinx <https://www.sphinx-doc.org/en/master/>`_, `myst-nb <https://myst-nb.readthedocs.io/en/latest/>`_, and `sphinx-book-theme <https://sphinx-book-theme.readthedocs.io/en/stable/>`_ with ``pip``. 
You can then generate the documentation by moving to the :orange:`/Docs` folder and executing::

    ./Make_doc 
     
The documentation will be generated in the :orange:`/Docs/build/html/` folder, and can be checked by opening the :orange:`index.html` file.





Secondary packages
------------------

A number of packages are required to run ``ANTARESS``. They should be installed automatically, but if you encounter some trouble you can install them manually as described below.

- Standard packages
    Install `arviz <https://python.arviz.org/en/stable/>`_, `astropy <https://www.astropy.org/>`_, `batman-package <https://lkreidberg.github.io/batman/docs/html/index.html>`_, `dace_query <https://dace.unige.ch/dashboard/>`_, `emcee <https://emcee.readthedocs.io/en/stable/>`_, `lmfit <https://lmfit.github.io/lmfit-py/>`_, `pandas <https://pandas.pydata.org/>`_, 
    `pathos <https://pathos.readthedocs.io/en/latest/pathos.html>`_, `PyAstronomy <https://pyastronomy.readthedocs.io/en/latest/>`_, `scipy <https://scipy.org/>`_, `statsmodels <https://www.statsmodels.org/stable/index.html>`_ using::
    
        pip install package         

- Resampling package 
    - `bindensity documentation <https://obswww.unige.ch/~delisle/bindensity/doc/>`_
    - install as a standard package::
    
        pip install bindensity

    - do not use ``bindensity`` with non-continuous grids, as it will mess up with banded covariance matrices.
    - beware when masking spectral ranges with ``ANTARESS``, as undefined pixels (set to nan values) are propagated by ``bindensity`` when resampling and will spread throughout the workflow.

- Package ``pySME`` 
    - follow these instructions to install ``PySME`` on M1/M2 Macs.
        - install rosetta by running::
        
            softwareupdate --install-rosetta

        - install Homebrew under rosetta by running::

            $ arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

          This installs Homebrew under :orange:`/usr/local/bin/brew` instead of the default for arm64 :orange:`/opt/homebrew/bin/brew`

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

    - set up the path to your local python installation in the `setup_lbl_fit.py <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_conversions/KitCat/setup_lbl_fit.py>`_ and run::
    
        python setup_lbl_fit.py build
        
      Then copy the compiled file :orange:`calculate_RV_line_by_line3.cpython-XX-darwin.so` into your `KitCat/ <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_conversions/KitCat/>`_ directory.  
