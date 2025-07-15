import h5py
import json
import os
import matplotlib.pyplot as plt
from glob import glob
import yaml


def create_config_from_template(template_file, output_file, **overrides):
    """
    Create a config file from a template with overrides.
    
    Args:
        template_file (str): Path to the template YAML file
        output_file (str): Path where the new config file will be saved
        **overrides: Keyword arguments for parameter overrides using dot notation
                    (e.g., "simulation.Lx"=80.0, "utility.args_A.k_ij"=-0.5)
    
    Returns:
        str: Path to the created config file
    
    Example:
        create_config_from_template(
            "configs/template.yaml",
            "configs/my_config.yaml",
            **{
                "simulation.Lx": 80.0,
                "simulation.Nx": 128,
                "utility.args_A.k_ij": -0.5,
                "utility.args_B.k_ij": -0.5
            }
        )
    """
    # Load template
    with open(template_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides using dot notation
    for key, value in overrides.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    # Save modified config
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_file


def load_data(filename):
    with h5py.File(filename, "r") as d:
        ϕA = d['tasks']["phiA"][()]
        ϕB = d['tasks']["phiB"][()]
        x = d['tasks']['phiA'].dims[1][0][()]
        t = d['tasks']['phiA'].dims[0]["sim_time"][()]
    return ϕA, ϕB, x, t

def load_config(config_file):
    """Load configuration from YAML file with defaults."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults for missing parameters
    with open("configs/template.yaml", "r") as f:
        defaults = yaml.safe_load(f)
    
    # Merge with defaults
    def merge_dicts(default, user):
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    config = merge_dicts(defaults, config)
    return config


def plot_kymo(datapath, save=True):
    files = glob(os.path.join(datapath, "*h5"))

    # load params
    with open(os.path.join(datapath, "params.json"), "r") as p:
        params = json.load(p)

    fig, ax = plt.subplots(1, 3, dpi=150,
                           figsize=(9, 4),
                           sharey=True,
                           sharex=True)
    for file in files:
        ϕA, ϕB, x, t = load_data(file)
        a0 = ax[0].pcolormesh(x, t, ϕA,
                              cmap="Blues",
                              vmin=0, vmax=1,
                              rasterized=True)
        a1 = ax[1].pcolormesh(x, t, ϕB,
                              cmap="Reds",
                              vmin=0, vmax=1,
                              rasterized=True)
        a2 = ax[2].pcolormesh(x, t, ϕA - ϕB,
                              cmap="RdBu_r",
                              vmin=-1, vmax=1,
                              rasterized=True)

    cax0 = ax[0].inset_axes([1.05, 0.0, 0.05, 1])
    cbar0 = fig.colorbar(a0, cax=cax0, ax=ax[0])
    cbar0.ax.set(title=r"$\phi^A$")
    ax[0].set(xlabel=r"$x$", ylabel=r"$t$")
    cax1 = ax[1].inset_axes([1.05, 0.0, 0.05, 1])
    cbar1 = fig.colorbar(a1, cax=cax1, ax=ax[1])
    cbar1.ax.set(title=r"$\phi^B$")
    ax[1].set(xlabel=r"$x$")
    cax2 = ax[2].inset_axes([1.05, 0.0, 0.05, 1])
    cbar2 = fig.colorbar(a2, cax=cax2, ax=ax[2])
    cbar2.ax.set(title=r"$\phi^A - \phi^B$")
    ax[2].set(xlabel=r"$x$")
    plt.tight_layout()

    if save:
        filename = os.path.basename(datapath) + "_kymo.pdf"
        fig.savefig(
            os.path.join(datapath, filename),
            bbox_inches="tight",
            dpi=300
        )

    return fig, ax