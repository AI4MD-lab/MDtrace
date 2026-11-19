"""Setup for mdtrace — Molecular Dynamics Trace."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
VERSION = {}
exec((ROOT / "mdtrace" / "version.py").read_text(encoding="utf-8"), VERSION)

setup(
    name="mdtrace",
    version=VERSION["__version__"],
    description="Molecular Dynamics Trace — trace the physics inside your MD trajectory",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ting Liang",
    author_email="liangting.zj@gmail.com",
    url="https://github.com/Tingliangstu/mdtrace",
    project_urls={
        "Documentation": "https://mdtrace.readthedocs.io/",
        "Source": "https://github.com/Tingliangstu/MDtrace",
        "Issues": "https://github.com/Tingliangstu/MDtrace/issues",
    },
    license="GPL-3.0-or-later",
    license_files=["LICENSE"],
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=[
        "numpy",
        "scipy",
        "netCDF4",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "mdtrace=mdtrace.main:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
