"""Explicit runtime configuration; importing this module does not change the environment."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def configure_runtime():
    """Apply user-selected compiler settings before importing CUDA libraries.

    Existing CUDA_HOME, CC, CXX, NVCC_CCBIN, and TILELANG_CACHE_DIR take precedence.
    Set STEREODETECT_CUDA_HOME or STEREODETECT_CACHE_DIR to opt into overrides.
    """
    cuda_home = os.environ.get("STEREODETECT_CUDA_HOME")
    if cuda_home:
        root = Path(cuda_home).expanduser().resolve()
        os.environ.setdefault("CUDA_HOME", str(root))
        for variable, executable in (
            ("CC", "x86_64-conda-linux-gnu-gcc"),
            ("CXX", "x86_64-conda-linux-gnu-g++"),
            ("NVCC_CCBIN", "x86_64-conda-linux-gnu-g++"),
        ):
            compiler = root / "bin" / executable
            if compiler.is_file():
                os.environ.setdefault(variable, str(compiler))
    cache = os.environ.get("STEREODETECT_CACHE_DIR")
    if cache:
        os.environ.setdefault("TILELANG_CACHE_DIR", str(Path(cache).expanduser().resolve()))


def package_versions(*names):
    """Record missing optional packages without making them runtime requirements."""
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions
