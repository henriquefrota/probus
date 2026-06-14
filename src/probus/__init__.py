"""Probus — model risk checks for quantitative research."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version declared in pyproject.toml,
    # read from the installed package metadata.
    __version__ = version("probus")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
