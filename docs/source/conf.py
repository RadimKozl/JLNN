import os
import sys
# Important: Edit the path to point to the 'jlnn' source folder
sys.path.insert(0, os.path.abspath('../../')) 

project = 'JLNN'
copyright = '2026, Ing. Radim Közl'
author = 'Ing. Radim Közl'
release = 'v0.0.2'

extensions = [
    'sphinx.ext.autodoc',       # For automatic generation from docstrings
    'sphinx.ext.napoleon',      # For parsing Google-style docstrings
    'sphinx.ext.viewcode',      # Adds a link to the source code
    'sphinx.ext.mathjax',       # For rendering LaTeX equations
    'sphinx_design',            # For modern grids and cards
]

# I recommend a more modern theme than alabaster (like furo or sphinx_rtd_theme)
html_theme = 'furo'