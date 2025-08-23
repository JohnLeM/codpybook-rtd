# Copyright (C)

import os
from distutils.core import setup

from setuptools import find_packages, setup

__version__ = "0.1.0"

DISTNAME = "codpybook"
DESCRIPTION = "A gallery of illustrative examples for the book 'Reproducing kernel methods for machine learning, PDEs, and statistics with Python'."
MAINTAINER = "jean-marc mercier"
MAINTAINER_EMAIL = "jeanmarc.mercier@gmail.com"
URL = "https://github.com/JohnLeM/codpybook-rtd"
LICENSE = "new BSD"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/JohnLeM/codpybook-rtd/issues",
    "Documentation": "https://codpybook-read-the-docs.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/JohnLeM/codpybook-rtd",
}

# codpy_path = os.path.dirname(__file__)
# codpy_path = os.path.join(codpy_path, "codpy")


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


# extra_files = package_files(codpy_path)
long_description = open("README.md", "r").read()

setup(
    name=DISTNAME,
    version=__version__,
    author=MAINTAINER,
    maintainer=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    # package_dir={"": "src"},
    # packages=find_packages(where="src"),
    # include_package_data=True,
    # package_data={"": extra_files},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Win32 (MS Windows)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    install_requires=[
        "codpy==0.2.1",
        "Sphinx",
        "sphinx-gallery",
        "sphinx-togglebutton==0.3.2",
        "sphinx-math-dollar==1.2.1",
        "ipython",
        "sphinx-rtd-theme==3.0.2",
        "xgboost",
        "seaborn==0.13.2",
        "tqdm==4.67.1",
        "QuantLib==1.39",
        "arch==7.2.0",
        "numdifftools==0.9.41",
        "scikit-image",
        "torchvision==0.14.0",
        "kagglehub==0.3.12",
        "gymnasium"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
