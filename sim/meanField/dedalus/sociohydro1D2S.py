import numpy as np
import dedalus.public as d3
import os
import argparse
import shutil
import warnings
import logging
from abc import ABC, abstractmethod
import yaml
from utils import load_config

logger = logging.getLogger(__name__)


class Utility(ABC):
    """
    Abstract base class for utility functions.
    """
    @abstractmethod
    def pi(self, phi_i, phi_j):
        pass
    @abstractmethod
    def dpii_dphii(self, phi_i, phi_j):
        pass
    @abstractmethod
    def dpii_dphij(self, phi_i, phi_j):
        pass

class QuadraticUtility(Utility):
    """
    Quadratic utility function:
    π_i = k_ii * φ_i + k_ij * φ_j + v_iii * φ_i^2 + v_iij * φ_i * φ_j + v_ijj * φ_j^2.
    """
    
    def __init__(self, k_ii, k_ij, v_iii, v_iij, v_ijj):
        """
        Initialize linear utility function.
        
        Args:
            k_ii: coefficient for self-interaction (∂π_i/∂φ_i)
            k_ij: coefficient for cross-interaction (∂π_i/∂φ_j)
            v_iii: coefficient for self-interaction (∂π_i/∂φ_i)
            v_iij: coefficient for cross-interaction (∂π_i/∂φ_j)
            v_ijj: coefficient for cross-interaction (∂π_i/∂φ_j)
        """
        self.k_ii = k_ii
        self.k_ij = k_ij
        self.v_iii = v_iii
        self.v_iij = v_iij
        self.v_ijj = v_ijj
    
    def pi(self, phi_i, phi_j):
        """π_i = k_ii * φ_i + k_ij * φ_j + v_iii * φ_i^2 + v_iij * φ_i * φ_j + v_ijj * φ_j^2"""
        return self.k_ii * phi_i + self.k_ij * phi_j + self.v_iii * phi_i**2 + self.v_iij * phi_i * phi_j + self.v_ijj * phi_j**2
    
    def dpii_dphii(self, phi_i, phi_j):
        """∂π_i/∂φ_i = k_ii + 2*v_iii*φ_i + v_iij*φ_j"""
        return self.k_ii + 2 * self.v_iii * phi_i + self.v_iij * phi_j
    
    def dpii_dphij(self, phi_i, phi_j):
        """∂π_i/∂φ_j = k_ij + v_iij*φ_i + 2*v_ijj*φ_j"""
        return self.k_ij + self.v_iij * phi_i + 2 * self.v_ijj * phi_j


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sociohydro mean field simulation with configurable utility functions"
    )
    parser.add_argument("config_file", type=str,
                        help="YAML configuration file")
    parser.add_argument("--savepath", type=str, default=None,
                        help="Override savepath from config file")

    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config_file)
    
    # Override savepath if specified
    if args.savepath:
        config['simulation']['savepath'] = args.override_savepath

    # simulation parameters
    Lx = config['simulation']['Lx']
    Nx = config['simulation']['Nx']
    t_stop = config['simulation']['t_stop']
    Δt = config['simulation']['dt']
    dealias = config['simulation']['dealias']
    sim_dt = config['simulation']['save_dt']
    timestepper = d3.SBDF2  # default
    dtype = np.float64  # default

    # sociohydro parameters
    T = config['sociohydro']['T']
    Γ = config['sociohydro']['Gamma']
    ϕA0 = config['sociohydro']['fillA']
    ϕB0 = config['sociohydro']['fillB']
    
    # Create utility functions
    utility_A = QuadraticUtility(**config["utility"]["args_A"])
    utility_B = QuadraticUtility(**config["utility"]["args_B"])

    # check stability
    Δx = Lx / Nx
    if Δt * Δx**(-4) > 1 / 8:
        warnings.warn("von Neumann stability not satisfied")

    # 1D basis
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    xbasis = d3.RealFourier(coord, Nx, bounds=(-Lx / 2, Lx / 2), dealias=dealias)

    # fields
    ϕA = dist.Field(name="phiA", bases=(xbasis))
    ϕB = dist.Field(name="phiB", bases=(xbasis))

    # problem
    problem = d3.IVP([ϕA, ϕB], namespace=locals())

    # substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    # vacancy
    ϕ0 = 1 - ϕA - ϕB
    # mobilities
    μA = ϕA * ϕ0
    μB = ϕB * ϕ0

    # utility functions
    
    # πA, dπA/dϕA, and dπA/dϕB
    πA = utility_A.pi(ϕA, ϕB)
    dπA_dϕA = utility_A.dpii_dphii(ϕA, ϕB)
    dπA_dϕB = utility_A.dpii_dphij(ϕA, ϕB)
    # πB, dπB/dϕA, and dπB/dϕB
    πB = utility_B.pi(ϕB, ϕA)
    dπB_dϕA = utility_B.dpii_dphij(ϕB, ϕA)
    dπB_dϕB = utility_B.dpii_dphii(ϕB, ϕA)


    ### Sociohydro ####
    # add equation
    problem.add_equation("dt(ϕA) - T*dx(dx(ϕA)) = T*(ϕA * dx(dx(ϕB)) - ϕB * dx(dx(ϕA))) - dx(μA * dx(πA)) - dx(Γ * μA * dx(dx(dx(ϕA))))")
    problem.add_equation("dt(ϕB) - T*dx(dx(ϕB)) = T*(ϕB * dx(dx(ϕA)) - ϕA * dx(dx(ϕB))) - dx(μB * dx(πB)) - dx(Γ * μB * dx(dx(dx(ϕB))))")
    #################

    ### solver and stopping ###
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = t_stop
    #################

    ### initial condition ###
    δϕA = 0.05
    δϕB = 0.05
    randA = np.random.uniform(low=-1, high=+1, size=Nx)
    randA -= randA.mean()
    randB = np.random.uniform(low=-1, high=+1, size=Nx)
    randB -= randB.mean()
    ϕA["g"] = ϕA0 + δϕA * randA
    ϕB["g"] = ϕB0 + δϕB * randB
    #################

    ### storage ###
    savepath = config['simulation']['savepath']
    os.makedirs(savepath, exist_ok=True)
    shutil.rmtree(savepath, ignore_errors=True)
    snapshots = solver.evaluator.add_file_handler(savepath,
                                                  sim_dt=sim_dt,
                                                  max_writes=1000)
    snapshots.add_task(ϕA, layout="g", name="phiA")
    snapshots.add_task(ϕB, layout="g", name="phiB")

    # store parameters
    params = config.copy()  # Store the entire config
    #################

    # main loop
    try:
        logger.info('Starting main loop...')
        while solver.proceed:
            solver.step(Δt)

            # nan check
            if np.isnan(np.min(ϕA["g"])) or np.isnan(np.min(ϕB["g"])):
                raise RuntimeError("got NaNs. Ending main loop.")

            # progress logger
            if solver.iteration % 10000 == 0:
                time_spent = solver.get_wall_time() - solver.start_time
                logger.info(
                    f"Iteration={solver.iteration}, "
                    f"sim time={solver.sim_time:0.2f} / {t_stop}, "
                    f"real Δt ={time_spent:0.2f} s")
    except:
        logger.error("Exception raised, ending main loop.")
        raise
    finally:
        solver.log_stats()
        with open(os.path.join(savepath, "config.yaml"), "w") as p:
            yaml.dump(params, p)
