# MORALS:  *Mo*rse Graph-aided discovery of *R*egions of *A*ttraction in a learned *L*atent *S*pace 

**MORALS: Analysis of High-Dimensional Robot Controllers via Topological Tools in a Latent Space**

Ewerton R. Vieira*, Aravind Sivaramakrishnan*, Sumanth Tangirala, Edgar Granados, Konstantin Mischaikow, Kostas E. Bekris

## Introduction
_MORALS_ combines autoencoding neural networks with Morse Graphs. It first projects the dynamics of the controlled system into a learned latent space. Then, it constructs a reduced form of Morse Graphs representing the bistability of the underlying dynamics, i.e., detecting when the controller results in a desired versus an undesired behavior.

## Installation
```
pip install MORALS
```

## Usage

### Reproduce experiments from the paper
1. Download the [Pendulum (LQR) dataset](https://drive.google.com/file/d/1C2SgOQiMpAkpjD-_WJykARZnUYduaL02/view?usp=sharing) from Google Drive.
2. Extract and place it inside `examples/data/`. There should be a directory `pendulum_lqr1k` and a labels file `pendulum_lqr1k_success.txt`.
3. Train the autoencoder and latent dynamics network: `python train.py --config pendulum_lqr.txt`.
4. Obtain the Morse Graph and the Regions of Attraction (RoAs) for the learned latent space dynamics: `python get_MG_RoA.py --config pendulum_lqr.txt --name_out pendulum_lqr --RoA --sub 16`.

### Try out MORALS on your own dataset


## Bibtex
If you find this repository useful in your work, please consider citing:
```
@misc{morals2023,
      title={${\tt MORALS}$: Analysis of High-Dimensional Robot Controllers via Topological Tools in a Latent Space}, 
      author={Ewerton R. Vieira and Aravind Sivaramakrishnan and Sumanth Tangirala and Edgar Granados and Konstantin Mischaikow and Kostas E. Bekris},
      year={2023},
      eprint={2310.03246},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
