# Project CS: propagating Starlink satellites
Created by: Emmanuel Mukeh, Justin Wong and Arjan van Staveren

A student project created for the course Computational Science at the University of Amsterdam. Not intended for practical use. \
⚠️ Note: this repo uses the python pickle module, which is not secure when loading files. Only load the included files in the runs folder. ⚠️

## How to run
Create a conda environment and install the required libraries by running the following commands:
```
conda create -n "[ANY NAME]" python=3.10.13
conda install -c conda-forge poliastro
pip install vtk
pip install sgp4
```
poliastro: https://docs.poliastro.space/en/stable/installation.html \
Used to convert OMM satellite data into position and velocity vectors, and propagating satellites with the effects of perturbations.

vtk: https://examples.vtk.org/site/Python/ \
Used to create a interactive 3d visualisation.

## Run a simulation
Running final_model.py will start a simulation of the satellites, and afterwards show a visualisation of them as well as a histogram. It also generates a file that contains the data of said distance histogram. Using the given settings, a single run takes 3600 timesteps (where every step, a second is simulated). This takes around 10 minutes. Model parameters that can be changed to decrease this time are: `simulation_length` and `max_count`. These variables can be found using ctrl+f. 

The chance of thruster failure, the independant variable in this research, can be found by searching for `perturbation_chance=`, which is actually the inverse of the thruster failure chance (a perturbation chance of 1.0 = thruster failure rate of 0.0).

## Review the data
Files that were generated using the model are loaded in the histogram.ipynb notebook. The contents are used to create the graphs.
