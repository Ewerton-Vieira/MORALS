from setuptools import setup

setup (
    name = "MORALS",
    version = "0.1.1",
    author = "Aravind Sivaramakrishnan, Dhruv Metha Ramesh, Ewerton Rocha Vieira",
    url = "https://github.com/Ewerton-Vieira/MORALS.git",
    description = "MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space",
    long_description = open('README.md').read(),
    ext_package='MORALS',
    packages=['MORALS'],
    install_requires = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas', 'seaborn', 'tqdm', 'torch', 'torchvision', 'wandb', 'CMGDB', 'dytop'],
)