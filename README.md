# Sociohydrodynamics: Data-Driven Modeling of Social Behavior

This repository contains code for data-driven modeling of social behavior using sociohydrodynamic approaches, associated with research published in [PNAS](https://doi.org/10.1073/pnas.2508692122) and [arXiv:2312.17627](https://doi.org/10.48550/arXiv.2312.17627).

## Overview

Sociohydrodynamics applies hydrodynamic principles to social phenomena, combining data-driven inference with active matter physics to model and understand population dynamics, segregation patterns, and social interactions.

## Repository Structure

- **`inference/`** - Linear regression tools for inferring sociohydrodynamic parameters from data
- **`ml/`** - Neural network models to learn US population dynamics
- **`sim/`** - Deterministic and stochastic simulations of the sociohydrodynamic PDEs and Schelling-like agent-based models.

Each folder contains its own `README` file describing their contents.

## Getting Started

1. Install dependencies: `conda env create -f sociohydro_environment.yml`
2. Activate environment: `conda activate sociohydro`
3. Explore notebooks in `ml/` directories for examples

## Citation

If you use this code, please cite the paper at [PNAS](https://doi.org/10.1073/pnas.2508692122).

## License

See [LICENSE](LICENSE) file for details.
