import os
import sys
from setuptools import setup, find_packages

setup (
    name = "MORALS",
    version = "0.1",
    packages = find_packages(),
    author = "Aravind Sivaramakrishnan, Dhruv Metha Ramesh, Ewerton Rocha Vieira",
    url = "https://github.com/Ewerton-Vieira/MORALS.git",
    description = "MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space",
    long_description = open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas', 'seaborn', 'tqdm', 'torch', 'torchvision', 'wandb', 'CMGDB'],
)