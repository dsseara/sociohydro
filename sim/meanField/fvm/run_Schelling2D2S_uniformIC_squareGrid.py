import fipy as fp
from fipy.tools import numerix
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nx", type=int, default=40,
                        help="number of cells in x")
    parser.add_argument("-dx", type=float, default=0.25,
                        help="cell size in x")
    parser.add_argument("-ny", type=int, default=40,
                        help="system size in y")
    parser.add_argument("-dy", type=float, default=0.25,
                        help="cell size in y")
    parser.add_argument("-duration", type=float, default=1000.0,
                        help="total duration of simulation")
    parser.add_argument("-k0", type=float, default=1.0,
                        help="self-utility of group 0")
    parser.add_argument("-k1", type=float, default=1.0,
                        help="self-utility of group 1")
    parser.add_argument("-kp", type=float, default=0.0,
                        help="symmetric cross-utility")
    parser.add_argument("-km", type=float, default=0.0,
                        help="antisymmetric cross-utility")
    parser.add_argument("-phi0", type=float, default=0.25,
                        help="initial uniform state of group 0")
    parser.add_argument("-phi1", type=float, default=0.25,
                        help="initial uniform state of group 1")
    parser.add_argument("-temp", type=float, default=0.1,
                        help="temperature")
    args = parser.parse_args()

    # model parameters
    κ0 = args.k0
    κp = args.kp
    κm = args.km
    κ1 = args.k1
    κ = np.array([[κ0, κp - κm],
                  [κp + κm, κ1]])
    Γ0 = 1  # surface tension of 0
    Γ1 = 1  # surface tension of 1
    ϕ0_init = args.phi0  # uniform state of 0
    ϕ1_init = args.phi1  # uniform state of 1
    
    temp = args.temp

    # simulation parameters
    nx = args.nx
    ny = args.ny
    dx = args.dx
    dy = args.dy
    duration = args.duration

    mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    
    ϕ0 = fp.CellVariable(name=r"$\phi_0$", mesh=mesh)
    μ̃0 = fp.CellVariable(name=r"$\tilde{\mu}_0$", mesh=mesh)
    ϕ1 = fp.CellVariable(name=r"$\phi_1$", mesh=mesh)
    μ̃1 = fp.CellVariable(name=r"$\tilde{\mu}_1$", mesh=mesh)

    noise0 = fp.GaussianNoiseVariable(mesh=mesh, mean=ϕ0_init,
                                      variance=0.0001).value
    noise1 = fp.GaussianNoiseVariable(mesh=mesh, mean=ϕ1_init,
                                      variance=0.0001).value
    ϕ0[:] = noise0
    ϕ1[:] = noise1

    M0 = ϕ0 * (1 - ϕ0 - ϕ1)
    π0 = κ[0, 0] * ϕ0 + κ[0, 1] * ϕ1
    μ0 = -π0 + temp * (np.log(ϕ0) - np.log(1 - ϕ0 - ϕ1))
    dμ0dϕ0 = -κ[0, 0] + temp * (1 - ϕ1) / (ϕ0 * (1 - ϕ0 - ϕ1))
    
    M1 = ϕ1 * (1 - ϕ0 - ϕ1)
    π1 = κ[1, 0] * ϕ0 + κ[1, 1] * ϕ1
    μ1 = -π1 + temp * (np.log(ϕ1) - np.log(1 - ϕ0 - ϕ1))
    dμ1dϕ1 = -κ[1, 1] + temp * (1 - ϕ0) / (ϕ1 * (1 - ϕ0 - ϕ1))
    
    eq0_1 = (fp.TransientTerm(var=ϕ0) == fp.DiffusionTerm(coeff=M0, var=µ̃0))
    
    eq0_2 = (fp.ImplicitSourceTerm(coeff=1, var=µ̃0)
             == fp.ImplicitSourceTerm(coeff=dμ0dϕ0, var=ϕ0)
             - dμ0dϕ0 * ϕ0 + μ0
             - fp.DiffusionTerm(coeff=Γ0, var=ϕ0))
    
    eq1_1 = (fp.TransientTerm(var=ϕ1) == fp.DiffusionTerm(coeff=M1, var=µ̃1))
    
    eq1_2 = (fp.ImplicitSourceTerm(coeff=1, var=µ̃1)
             == fp.ImplicitSourceTerm(coeff=dμ1dϕ1, var=ϕ1)
             - dμ1dϕ1 * ϕ1 + μ1
             - fp.DiffusionTerm(coeff=Γ1, var=ϕ1))

    eq = eq0_1 & eq0_2 & eq1_1 & eq1_2

    dexp = -5
    
    nt = 100
    ϕ_array = np.zeros((2, nt+1, nx ,ny))
    ϕ_array[0, 0] = np.reshape(ϕ0, (nx, ny))
    ϕ_array[1, 0] = np.reshape(ϕ1, (nx, ny))
    t_save = np.linspace(0, duration, nt+1)
    elapsed = 0
    
    flag = 0
    while elapsed < duration:
        dt = min(100, np.exp(dexp))
        elapsed += dt
        dexp += 0.01
        eq.solve(dt=dt)
        print(f"t={elapsed:0.2f} / {duration}", end="\r")
        if elapsed >= t_save[flag]:
            ϕ_array[0, flag] = np.reshape(ϕ0, (nx, ny))
            ϕ_array[1, flag] = np.reshape(ϕ1, (nx, ny))
            flag += 1

    # dϕ0dy, dϕ0dx = np.gradient(ϕ_array[0], dy, dx, axis=(1, 2))
    # dϕ1dy, dϕ1dx = np.gradient(ϕ_array[1], dy, dx, axis=(1, 2))
