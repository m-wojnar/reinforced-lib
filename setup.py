from typing import List

from setuptools import setup, find_packages


def read_requirements(filename: str) -> List[str]:
    with open(filename) as file:
        return file.read().splitlines()


basic_requirements = read_requirements('requirements/requirements.txt')
test_requirements = read_requirements('requirements/requirements-test.txt')
examples_requirements = read_requirements('requirements/requirements-test.txt')
docs_requirements = read_requirements('requirements/requirements-test.txt')


setup(
    name='reinforced-lib',
    version='0.1.0',
    packages=find_packages(include=[
        'reinforced_lib', 
        'reinforced_lib.*'
    ]),
    install_requires=basic_requirements,
    tests_require=test_requirements,
    extras_require={
        'dev': basic_requirements + test_requirements + examples_requirements + docs_requirements
    }
)
