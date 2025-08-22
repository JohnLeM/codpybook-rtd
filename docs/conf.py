import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "..")))

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)

# ---- Base Extensions -------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_togglebutton",
    "sphinx_math_dollar",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.viewcode",
    'IPython.sphinxext.ipython_console_highlighting',
]
autosummary_generate = True
autoclass_content = 'both'
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,  # This line tells Sphinx to skip undocumented members
    "private-members": False,  # This line tells Sphinx to skip private members
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

def skip_undoc_members(app, what, name, obj, skip, options):
    if not skip and (not hasattr(obj, '__doc__') or not obj.__doc__):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_undoc_members)



mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for Sphinx-Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "ch7",
        "ch8",
        "ch9",
    ],  # Directory where .py scripts are stored
    "gallery_dirs": [
        "auto_ch2",
        "auto_ch3",
        "auto_ch4",
        "auto_ch5",
        "auto_ch6",
        "auto_ch7",
        "auto_ch8",
        "auto_ch9",
    ],  # Directory to save generated HTML and notebooks
    "ignore_pattern": r"ignore.*\.py",  
    "filename_pattern": r".*\.py",  # Process all Python files
    # "ignore_pattern": r"(data_gen|1NN_estimation_rate|multiscale_OT|mini_sinkhorn|Bachelier|MNIST|MultiscaleMNIST)\.py",
    "backreferences_dir": "gen_modules/backreferences",
    "within_subsection_order": "ExampleTitleSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "notebook_images": True,  # Include images in generated notebooks
    "image_scrapers": ("matplotlib"),
    "matplotlib_animations": True
    
}

project = "CodPy Book"
author = "Philippe G. LeFloch , Jean-Marc Mercier, and Shohruh Miryusupov"
copyright = "2024, Philippe G. LeFloch , Jean-Marc Mercier, and Shohruh Miryusupov"
release = "1.0.0"


# Generate summary tables automatically
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# Set the theme to "sphinx_rtd_theme"
html_theme = "sphinx_rtd_theme"

# Additional theme options for further customization
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "style_external_links": True,
    "titles_only": False,
}

# Add paths that contain custom static files (such as CSS files) here, relative to this directory.
# html_static_path = ["_static"]

# Custom CSS for further styling
html_css_files = [
    "custom.css",  # Add your custom CSS file for extra styling if needed
]

# -- Options for Napoleon ---------------------------------------------------

# Napoleon settings for NumPy and Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- MathJax options (for rendering mathematical expressions) ----------------

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# -- Options for the documentation -------------------------------------------

# Enable the nitpicky mode to warn about all missing references
nitpicky = False

suppress_warnings = ["ref.python", "ref"]
# -- Version control ---------------------------------------------------------

# If you want to version the documentation, add this
html_show_sourcelink = True

# -- Localization ------------------------------------------------------------

# The language for content autogenerated by Sphinx
language = "en"
