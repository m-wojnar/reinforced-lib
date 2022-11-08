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
    license='Mozilla Public License 2.0 (MPL 2.0)',
    description='Reinforcement learning library',
    author='Maksymilian Wojnar and Wojciech Ciężobka',
    url='https://github.com/m-wojnar/reinforced-lib',
    download_url='https://github.com/m-wojnar/reinforced-lib/archive/refs/tags/v0.1.0.tar.gz',
    keywords='machine-learning, reinforcement-learning, reinforcement-learning-agent, jax',
    python_requires='>=3.8, <4',
    install_requires=read_requirements('requirements/requirements.txt'),
    extras_require={'dev': read_requirements('requirements/requirements-dev.txt')},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed'
    ],
)
