# Simulation Framework

This directory contains simulation tools for sociohydrodynamic modeling, including both deterministic and stochastic approaches.

## Directory Structure

### `meanField/`
- **`fvm/`** - Finite volume method implementations for solving sociohydrodynamic PDEs
  - `fvm_utils.py` - Utility functions for finite volume calculations
  - `run_Schelling2D2S_geographicArea_quadraticUtility_quadraticGrowth.py` - 2D two-species Schelling model with geographic constraints
- **`SchellingMF/`** - Julia-based mean field theory simulations
  - `run_SchellingMF.jl` - Main execution script for mean field simulations

### `monteCarlo/`
The Monte Carlo simulations are split according to the number of species and spatial dimension of each simulation. Each simulation 
- **`SchellingNDMS/`** - N-Dimensional, M-species Schelling model
  - `run_SchellingNDMS.jl` - Basic Monte Carlo simulation
  - `run_SchellingNDMSAnneal.jl` - Simulated annealing variant

## Usage

- **Python simulations**: Run directly with `python filename.py`
- **Julia simulations**: Use `julia filename.jl` or activate the project environment first

