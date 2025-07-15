# Generic Quadratic Utility Functions for Sociohydro Mean Field Simulation

This document explains how to use the generic quadratic utility function system in the Sociohydro mean field simulation with YAML configuration files.

## Overview

The equation of motion is given by
$$
\partial_t \phi_a(x, t) = \nabla \cdot \left( T\left( \phi_0 \nabla \phi_a - \phi_a \nabla \phi_0 \right) - \phi_a \phi_0 \nabla \pi_a - \Gamma \phi_a \phi_0 \nabla^3 \phi_a \right)
$$
where $\phi_a \in [0, 1]$ is the fill fraction of group $a \in [A, B]$. The factors of $\phi_0 = 1 - \sum_j \phi_j$ measure the vacancy fill fraction, and modulates the dynamics to take into account volume exclusion.
The first term describes diffusion (note that $\phi_0 \to 0$ leads to standard Fickian diffusion).
The second term describes advection with a velocity $\mathbf{v} = \nabla \pi^a(\vec{\phi})$, where the utility function $\pi_a(\vec{\phi})$ measures how much an agent of type $a$ likes an area with occupation configuration $\vec{\phi} = (\phi^A, \phi^B)$.
The third term acts as a surface tension, penalizing spatial gradients in the fields $\phi_a$.

The simulation uses a generic quadratic utility given by
$$
\pi_a =  \kappa_{ab} \phi_b + \nu_{abc} \phi_b \phi_c,
$$
where we use Einstein notation to imply summation over repeated indices.

This simulation uses Dedalus to solve the equations in 1D over periodic boundary conditions.

## Configuration File Structure

The simulation uses YAML configuration files with the following structure:

```yaml
simulation:
  Lx: 40.0          # System size
  Nx: 64            # Number of spatial points (should be power of 2 for FFT)
  t_stop: 50.0      # Total simulation time
  dt: 0.01          # Time step (check stability condition)
  dealias: 1.5      # Dealiasing factor for spectral methods
  save_dt: 1.0      # How often to save output
  savepath: "output_template"  # Output directory

sociohydro:
  Gamma: 1.0        # Stabilizing strength (higher = more stable)
  T: 0.1            # Temperature (plays role of diffusion constant)
  fillA: 0.25       # Initial fill fraction of agent A
  fillB: 0.25       # Initial fill fraction of agent B

utility:
  args_A:
    k_ii: 1.0     # Self-interaction linear
    k_ij: 0.0     # Cross-interaction linear
    v_iii: 0.0     # Self-interaction quadratic
    v_iij: 0.0     # Cross-interaction quadratic
    v_ijj: 0.0     # Other-interaction quadratic
  args_B:
    k_ii: 1.0     # Self-interaction linear
    k_ij: 0.0     # Cross-interaction linear
    v_iii: 0.0     # Self-interaction quadratic
    v_iij: 0.0     # Cross-interaction quadratic
    v_ijj: 0.0     # Other-interaction quadratic
```

## Running Simulations

### Basic Usage

```bash
python sociohydro1D2S_meanField.py config.yaml
```

## Example Configurations

### 1. Segregation
```yaml
utility:
  args:
    k_AA: 1.0
    k_AB: -0.5
    k_BB: 1.0
    k_BA: -0.5
    v_AAA: 0.0
    v_AAB: 0.0
    v_ABB: 0.0
    v_BBB: 0.0
    v_BBA: 0.0
    v_BAA: 0.0
```
Both agents prefer their own type over the other type (linear case).

### 2. Migration
```yaml
utility:
  args:
    k_AA: 1.0
    k_AB: -1.0
    k_BB: 1.0
    k_BA: 1.0
    v_AAA: 0.0
    v_AAB: 0.0
    v_ABB: 0.0
    v_BBB: 0.0
    v_BBA: 0.0
    v_BAA: 0.0
```
Agent A strongly prefers its own type, while agent B weakly prefers its own type.

## Parameter Sweeps

The example script demonstrates how to run parameter sweeps:

```python
# Create configs with varying segregation strength
for strength in [0.1, 0.3, 0.5, 0.7, 0.9]:
    config_file = f"configs/sweep_segregation_{strength}.yaml"
    create_config_from_template(
        "configs/segregation_linear.yaml",
        config_file,
        **{
            "utility.args.k_AB": -strength,
            "utility.args.k_BA": -strength,
            "simulation.savepath": f"output_sweep_segregation_{strength}"
        }
    )
    run_simulation(config_file)
```

## Creating Custom Utility Functions

The current implementation uses a generic quadratic utility function that can represent most common interaction patterns. If you need more complex utility functions, you can modify the `QuadraticUtility` class or create a new class that inherits from `UtilityFunction`.

To create your own utility function, inherit from `UtilityFunction` and implement the required methods:

```python
class MyCustomUtility(UtilityFunction):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def pi(self, phi_self, phi_other):
        """Compute utility π_i(φ_i, φ_j)"""
        return self.param1 * phi_self + self.param2 * phi_other**2
    
    def dpi_dphi_self(self, phi_self, phi_other):
        """Compute ∂π_i/∂φ_i"""
        return self.param1
    
    def dpi_dphi_other(self, phi_self, phi_other):
        """Compute ∂π_i/∂φ_j"""
        return 2 * self.param2 * phi_other
```

Then modify the main script to use your custom utility function and add corresponding YAML parameters.

## Running Examples

Use the provided example script to run multiple simulations:

```bash
python example_usage.py
```

This will run various scenarios demonstrating different utility function configurations.

## Output

The simulation saves:
1. HDF5 files with time series data
2. Parameter file (`params.json`) with the complete configuration
3. Kymograph plots showing the evolution of φ_A, φ_B, and φ_A - φ_B

## Advantages of YAML Configuration

1. **Readability**: Human-readable configuration format
2. **Modularity**: Easy to create and modify configurations
3. **Reproducibility**: Complete parameter sets saved with results
4. **Parameter Sweeps**: Easy to generate multiple configurations
5. **Version Control**: Configurations can be tracked in git
6. **Documentation**: Comments and structure make parameters self-documenting

## Notes

- The utility functions are evaluated at each spatial point and time step
- Derivatives are computed analytically for efficiency
- The system supports asymmetric utilities between agents A and B
- All parameters are saved in the output for reproducibility
- Missing parameters in YAML files are filled with sensible defaults
- The generic quadratic form reduces to linear when all quadratic coefficients (v_*) are set to zero
- The utility function can represent complex interactions including diminishing returns, competition, and cooperation 