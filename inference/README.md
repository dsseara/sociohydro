# Inference Framework

This directory contains tools for inferring sociohydrodynamic parameters from real-world data using linear regression and other inference methods.

## Directory Structure

### Core Inference Tools
- **`sociohydroInferer.py`** - Main class for inferring sociohydrodynamic parameters for data on a grid
- **`sociohydroInferer_fipy.py`** - Class for inferring sociohydrodynamic parameters for data on a mesh, using `FiPy`'s derivatives
- **`infer_utils.py`** - Utility functions for inference calculations and data processing

### Inference Scripts
- **`infer_sociohydro2D.py`** - 2D sociohydrodynamic parameter inference
- **`infer_sociohydro2D_nn.py`** - Neural network-assisted inference for 2D models
- **`infer_sociohydro2D_census.py` - Census data-based inference for 2D models

### Data and Testing
- **`census_dataset.py`** - Dataset handling for census data inference

## Usage

- **Python scripts**: Run directly with `python filename.py`
- **Main inference**: Use `sociohydroInferer.py` as the primary inference class