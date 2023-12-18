# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import glob

#Identify whether doc is generated on gitlab or locally
conf_path = os.path.abspath(os.path.dirname(__file__))
print(conf_path)
if conf_path=='/builds/bourrier/antaress/Docs/source':
	sys.path.insert(0, os.path.abspath('/builds/bourrier/antaress/Method/'))
	for antaress_dir in glob.glob('/builds/bourrier/antaress/Method/ANTARESS_*/'):
		sys.path.insert(0, os.path.abspath(antaress_dir))
else:
	sys.path.insert(0, os.path.abspath('/Users/bourrier/Travaux/ANTARESS/Method/'))
	for antaress_dir in glob.glob('/Users/bourrier/Travaux/ANTARESS/Method/ANTARESS_*/'):
		sys.path.insert(0, os.path.abspath(antaress_dir))


# -- Project information -----------------------------------------------------

project = 'ANTARESS'
copyright = '2023, Vincent Bourrier'
author = 'Vincent Bourrier'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [	'sphinx.ext.autodoc',
			'sphinx.ext.autosummary',
			'sphinx.ext.napoleon',
			'sphinx.ext.viewcode',
			'sphinx.ext.intersphinx',
			'sphinx.ext.githubpages',
			'sphinx.ext.mathjax',
			'myst_nb']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#Napoleon settings
napoleon_include_init_with_doc = True
napoleon_custom_sections = [('Returns', 'params_style')]

#Various settings
add_module_names = False  # Remove namespaces from class/method signatures
autodoc_member_order = "bysource"

#myst_nb options
myst_enable_extensions = ["dollarmath", "colon_fence"]
nb_execution_mode = "force"
nb_execution_timeout = -1

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
#html_theme_options = {
 #   "path_to_docs": "docs",
  #  "repository_url": repository_url,
   # "repository_branch": repository_branch,
    #"use_edit_page_button": False,
#    "use_issues_button": False,
 #   "use_repository_button": False,
 #   "use_download_button": False,
#}












