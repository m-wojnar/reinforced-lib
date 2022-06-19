from setuptools import setup, find_packages

setup(
    name='reinforced-lib',
    version='0.1.0',
    packages=find_packages(include=[
        'reinforced_lib', 
        'reinforced_lib.*'
    ])
)