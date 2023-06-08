import os

import setuptools

from masif import (
    author,
    author_email,
    description,
    package_name,
    # project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl_sphinx_theme",
        # Others
        "mypy",
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        "pre-commit",
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url=url,
    # project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "torch==1.12.1",
        "wandb==0.12.14",
        "tqdm==4.62.3",
        "absl-py==1.2.0",
        "hydra-core==1.2.0",
        "matplotlib==3.5.2",
        "tables",
        "scikit-learn",
        "torchsort==0.1.8",
        "pandas==1.4.2",
        "omegaconf==2.2.3",
        "yahpo_gym",
        "openml",
        "ConfigSpace",
        "smac==1.4.0",
        # "lcdb",
        # "pytorch_forecasting",
        # "pytables",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
