from setuptools import setup, find_packages


def read_requirements(filename: str) -> list[str]:
    with open(filename) as file:
        return file.read().splitlines()


def read_file(filename: str) -> str:
    with open(filename) as file:
        return file.read()


setup(
    name='reinforced-lib',
    version='0.2.0',
    packages=find_packages(include=[
        'reinforced_lib', 
        'reinforced_lib.*'
    ]),
    license='Mozilla Public License 2.0 (MPL 2.0)',
    description='Reinforcement learning library',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Maksymilian Wojnar and Wojciech Ciężobka',
    url='https://github.com/m-wojnar/reinforced-lib',
    download_url='https://github.com/m-wojnar/reinforced-lib/archive/refs/tags/v0.2.0.tar.gz',
    keywords='deep-reinforcement-learning, jax, library, machine-learning, reinforcement-learning',
    python_requires='>=3.9, <4',
    install_requires=read_requirements('requirements/requirements.txt'),
    extras_require={
        'docs': read_requirements('requirements/requirements-docs.txt'),
        'full': read_requirements('requirements/requirements-full.txt')
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed'
    ],
)
