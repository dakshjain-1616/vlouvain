"""
VLouvain package setup.
"""

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vlouvain",
    version="0.1.0",
    description=(
        "Cluster 1M vectors in under 1 second without building a graph — "
        "first pip-installable VLouvain implementation."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=os.getenv("VLOUVAIN_REPO_URL", "https://github.com/vlouvain/vlouvain"),
    author=os.getenv("VLOUVAIN_AUTHOR", "NEO"),
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "hdbscan": ["hdbscan>=0.8.33"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-timeout>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vlouvain=vlouvain.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
