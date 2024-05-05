import os
from typing import List

from setuptools import setup


def read_version() -> str:
    with open(os.path.join(os.path.dirname(__file__), "build.number"), mode="r") as file:
        version = file.read()
    return version


def read_requirements() -> List[str]:
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), mode="r") as file:
        requirements = [package.strip() for package in file.readlines()]
    return requirements


if __name__ == "__main__":
    setup(
        name="recerr",
        version=read_version(),
        packages=[
            "recerr",
            "recerr.explain",
            "recerr.utils",
            "recerr.metrics",
            "recerr.splitting",
        ],
        url="",
        license="MIT License",
        author="amtsyplov",
        author_email="",
        description="",
        install_requires=read_requirements(),
    )
