"""Sphinx configuration for ryd-gate documentation."""

project = "ryd-gate"
copyright = "2024, Siyuan Chen"
author = "Siyuan Chen"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
