# -*- coding: utf-8 -*-

# Copyright 2023-2024 Vincent Bourrier
#
# This file is part of antaress.
#
# antaress is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# antaress is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with antaress.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, Extension
import numpy as np
import os

path = os.path.abspath(os.path.dirname(__file__))

info = {}
with open(os.path.join(path, '__info__.py'), 'r') as f:
  exec(f.read(), info)
with open('README.md', 'r', encoding='utf-8') as readme:
  long_description = readme.read()

#c_ext = Extension('spleaf.libspleaf',
#  sources=['spleaf/pywrapspleaf.c', 'spleaf/libspleaf.c'],
#  language='c')

setup(name=info['__title__'],
  version=info['__version__'],
  author=info['__author__'],
  author_email=info['__author_email__'],
  license=info['__license__'],
  description=info['__description__'],
  long_description=long_description,
  url=info['__url__'],
  packages=['antaress'],
  #ext_modules=[c_ext],
  include_dirs=[np.get_include()],
  python_requires='>=3.6',
  install_requires=['numpy>=1.16', 'scipy>=1.3'])
