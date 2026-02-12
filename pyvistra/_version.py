"""Version helpers for pyvistra."""

from importlib.metadata import PackageNotFoundError, version


# Fallback version for source-tree runs when package metadata is unavailable.
__version__ = "0.3.0"


def get_version():
    """Return installed package version, falling back to source version."""
    try:
        return version("pyvistra")
    except PackageNotFoundError:
        return __version__
