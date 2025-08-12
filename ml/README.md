# Sociohydrodynamics: data-driven modelling of social behavior

This repository contains the code for training and analyzing machine learning models in [ArXiv:2312.17627](https://arxiv.org/abs/2312.17627). The **census_density_gridded** folder contains models trained using US Census data from 1980-2020 as well as saliency analyses of these models. This is the data to be used in the revision of the paper. The code for fipy simulations of coarse-grained residential dynamics is available at [this repository](https://github.com/dsseara/sociohydro). 

## Directory Organization
```
├── census_density_gridded	: Final neural network census models
	├── configs			: Training configuraiton files
	├── models			: Trained models and saliency computations
├── nonlinear_simulations	: Neural networks trained on nonlinear utility sims
	├── Figures			: Saved figures
	├── models			: Trained models and saliency computations
├── nonlocal_ising		: Neural networks trained on non-local Ising model sims
	├── Figures			: Saved figures
	├── models			: Trained models and saliency computations
├── sociohydro_environment.yml	: conda environment YAML file

------ LEGACY CODE ------
├── linear_simulations		: Neural networks trained on linear utility sims
├── census_density		: Legacy neural network code
├── decennial			: Legacy neural network code
├── param_inference		: Legacy parameter estimation code

```


The **nonlinear_simulations** folder contains models trained using coarse-grained data from simulations of an agent-based model [(Github)](https://github.com/dsseara/sociohydro). 
This is used to validate the methodology and demonstrate how network saliency can reflect utility functions. Similarly, the **nonlocal_ising** folder contains models trained using coarse-grained data from simulations of a Kawasaki dynamics ising model, also used to validate the methodology.

## Legacy code

In addition to the above folders, we include additional folders for legacy purposes. These formed the basis of preliminary investigations but the data should not be considered final. We are leaving these folders in this repository because they may have instructive value, reflecting the fact that much of research is discovering why things do not work the way you expect them to.

The **census_density** folder contains models trained using US census data. However, we realized that our coarse-graining scheme was incorrect as it did not preserve total population. This was fixed in the census_density_gridded folder. 

The **linear_simulations** folder contains models trained using coarse-grained data from simulations of agents with *linear* utility functions. This was used to validate the methodology, but we eventually moved to evaluating nonlinear utility functions.

The **decennial** folder also contains models trained using US Census data from 1980-2020. Due to the irregular geometries of each county, we adapted
the physical bottleneck method introduced in [this paper](https://www.cell.com/cell/fulltext/S0092-8674(23)01331-4) [(Github)](https://github.com/schmittms/physical_bottleneck).
The neural networks predict the time derivatives of US census data and these are used to evolve the system in time using Fenics/Dolfin. The network itself is trained using dolfin-adjoint. 
This approach to learning and predicting dynamics was used during the initial draft of the paper.
During the revision stage, we noticed potential issues with the meshing procedure applied to each US county, as well as the process for using the models to predict dynamics (the physical bottleneck method was previously only used to predict static behavior).
Because these concerns limited our confidence in drawing conclusions about the network saliencies, we changed our machine learning pipeline to that used in census_gridded.

The **param_inference** folder contains early attempts to infer parameters for the hydrodynamics models. In particular, the folder contains some attempts with neural networks to capture growth terms, as well as an MCMC pipeline that was never fully integrated with the fipy simulations. 
