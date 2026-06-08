"""Compatibility shim for editable installs.

Package metadata, dependencies, and Python-version constraints are defined in
``pyproject.toml``. Keeping this file minimal prevents legacy ``setup.py``
metadata from drifting out of sync with the release metadata.
"""

from setuptools import setup


setup()
