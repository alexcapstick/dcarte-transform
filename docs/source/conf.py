# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath("../.."))

# Mock dcarte because it fails to build in readthedocs
MOCK_MODULES = ["dcarte", "dcarte.local", "dcarte.utils"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

import dcarte_transform

version = dcarte_transform.__version__
doc = dcarte_transform.__doc__
author = dcarte_transform.__author__
project = dcarte_transform.__title__
copyright = dcarte_transform.__copyright__
release = '.'.join(version.split('.')[:2])

# -- General configuration

extensions = [
    'readthedocs_ext.readthedocs',
    'sphinx.ext.duration',
    "sphinx_rtd_theme",
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx_autodoc_typehints",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

master_doc = 'index'