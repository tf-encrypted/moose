# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyMoose'
copyright = '2023, TF Encrypted Authors'
author = 'TF Encrypted Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_remove_toctrees",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

autosummary_generate = True
napoleon_include_init_with_doc = True
myst_heading_anchors = 3
myst_enable_extensions = ['dollarmath']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    # "logo_only": True,
    "show_toc_level": 1,
    "repository_url": "https://github.com/tf-encrypted/moose",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

html_static_path = ['_static']
remove_from_toctrees = ["_autosummary/*"]

