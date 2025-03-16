# -*- coding: utf-8 -*-
import numpy as np
from setuptools import Extension, setup

setup(
  ext_modules=[
    Extension(
     'antaress.Gauss_star_grid',
      sources=['src/antaress/ANTARESS_analysis/C_grid/Gauss_star_grid.c'],
      language='c',
    )
  ],
  include_dirs=[np.get_include()],
)
