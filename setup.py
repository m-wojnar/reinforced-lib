from typing import List

from setuptools import setup, find_packages


def read_requirements(filename: str) -> List[str]:
    with open(filename) as file:
        return file.read().splitlines()


setup(
    name='reinforced-lib',
    version='0.1.0',
    packages=find_packages(include=[
        'reinforced_lib', 
        'reinforced_lib.*'
    ]),
    install_requires=read_requirements('requirements/requirements.txt'),
    tests_require=read_requirements('requirements/requirements-dev.txt'),
    extras_require={
        'dev': read_requirements('requirements/requirements-dev.txt')
    }
)
