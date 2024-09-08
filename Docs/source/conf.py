# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/antaress/'))

# -- Project information -----------------------------------------------------
project = 'ANTARESS'
copyright = '2023-2024, V. Bourrier'
author = 'V. Bourrier & contributors'

#Get release version
with open('../../pyproject.toml', 'r') as f:
    for line in f.readlines():
        if 'version' in line:
            release = line.split('= "')[1].split('"')[0]  
            break
    
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
numfig = True #Figure referencing

#myst_nb options
myst_enable_extensions = ["dollarmath", "colon_fence"]
nb_execution_mode = "force"
nb_execution_timeout = -1

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_theme_options = {
    "gitlab_url": "https://gitlab.unige.ch/bourrier/antaress",
    "logo": {
        "text": "Version "+release,
    },
    "navigation_with_keys":True,    
    }
html_logo = "./Fixed_files/antaress_webp.png"
html_favicon = "./Fixed_files/antaress_icon.ico"

html_sidebars = {
    "**": ["navbar-logo.html","search-field.html","sbt-sidebar-nav.html"]
}



