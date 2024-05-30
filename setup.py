from setuptools import setup, find_packages

setup (
    name = "MORALS",
    version = "0.1.3",
    author = "Aravind Sivaramakrishnan, Ewerton Rocha Vieira, Sumanth Tangirala",
    url = "https://github.com/Ewerton-Vieira/MORALS.git",
    description = "MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space",
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_package='MORALS',
    packages=find_packages(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas', 'seaborn', 'tqdm', 'torch', 'torchvision', 'CMGDB', 'dytop', 'tqdm'],
)