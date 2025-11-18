import os
import sys

sys.path.insert(0, os.path.abspath("../python"))

project = "Lacuna"
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = ["lacuna._core", "_core"]
html_theme = "pydata_sphinx_theme"
html_logo = "logo.png"
html_theme_options = {
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "TODO": ["sourcelink"],
        "changelog": ["sourcelink"],
        "develop": ["sourcelink"],
    }
}
pygments_dark_style = "monokai"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
myst_enable_extensions = [
    "tasklist",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
