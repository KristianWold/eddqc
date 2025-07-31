# Analysis

This folder contains all notebooks and data required to regenerate the results and figures in the paper.
To reproduce Figure 1, 2 and 3, the major figures of this work, run [plot.ipynb](plotting/plot.ipynb) located in the folder [plotting](plotting/).

## experiments

scripts and standalone source code for setting up integrable and non-integrable quantum circuits for various depths.

## fitting_maps

Retrieving quantum map models on the data produced by running the script in [experiments](experiments/) on the Zhejiang platform.
Raw experimental data is found in [fitting_maps/data/chaos_exp_reorder/5q/](fitting_maps/data/chaos_exp_reorder/5q/). 
Fitted models are found in [fitting_maps/models/](fitting_maps/models/).

## fitting_maps_bootstrap

Retrieving quantum map models on data resampled from [fitting_maps/data] using bootstrapping. Resulting models are in [fitting_maps_bootstrap/models/](fitting_maps_bootstrap/models/).

## integrable_nonintegrable

Data and scripts for analysis related to Figure 2.

## integrable_transition

Data and scripts for analysis related to Figure 3.

## models

Quantum map models retrieved from experimental quantum circuit data.

## plotting

Scripts for plotting Figure 1, 2 and 3.

## synthetic_benchmark 

Data and scripts for analysis related to Figure S6.

## theoretic_AI_and_FF

Data and scripts for analysis related to Figure 1.

