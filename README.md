# Experimental Detection of Dissipative Quantum Chaos

Code, data and analysis scripts for the paper: [Experimental Detection of Dissipative Quantum Chaos
](https://arxiv.org/pdf/2506.04325).

## Overview

This repository contains the source code, data, and analysis scripts for the paper "Experimental Detection of Dissipative Quantum Chaos", which explores detection of quantum chaos (and lack thereof) in noisy quantum ciruits on real quantum hardware. 

## Repository Structure
```
├── analysis/                                       # subfolders containing notebooks, data, and resulting figures
│   ├── experiments/
│   ├── fitting_maps/           
│   │   ├── data/                                   # raw experimental data from Zhejiang Platform
│   │   ├── models/                                 # quantum map models retrieved form data
│   │   ├── Fitting_Map_Integrable_L=5.ipynb        # notebooks for retriving quantum map models
│   │   └── ...
│   ├── fitting_maps_bootstrap/                     # quantum maps retrieved from bootstrapped data
│   └── ...
├── src/                                            # source code                 
│   ├── analysis.py
│   ├── experimental.py                   
│   └── ...
├── LICENSE.md
└── ...
```

## Requirements

Main libraries used are
- qiskit
- tensorflow
- matplotlib
- seaborn
- jupyter

For an exhaustive list, see [requirements.txt](requirements.txt) for Windows dependencies.

## Contact

personal email: kristian.wold@hotmail.com
university email: krisw@oslomet.no