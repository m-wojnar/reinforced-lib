from setuptools import setup, find_packages

setup(
    name='reinforced-lib',
    version='0.1.0',
    packages=find_packages(include=[
        'reinforced_lib', 
        'reinforced_lib.*'
    ]),
    install_requires=[
        'chex>=0.1.3',
        'distrax>=0.1.2',
        'dm-haiku>=0.0.7',
        'gym>=0.25.0',
        'jax>=0.3.15',
        'lz4>=4.0.2',
        'optax>=0.1.3',
        'scipy>=1.8.1',
    ],
    extras_require={
        'dev': [
            'matplotlib>=3.5.2',
            # 'ns3-ai==1.0.0',
            'seaborn>=0.11.2',
            'sphinx>=5.1.1',
            'sphinx-rtd-theme>=1.0.0',
        ]
    }
)
