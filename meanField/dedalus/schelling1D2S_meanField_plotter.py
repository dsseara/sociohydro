import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation, rc
from IPython.display import HTML
import h5py
import argparse
from glob import glob
import os
import json

# def make_movie(x, ϕA_list, ϕB_list):
#     fig, ax = plt.subplots(dpi=150, figsize=(4, 2))
#     # initialize line
#     x = dist.local_grid(xbasis, scale=dealias)
#     lineA, = ax.plot(x, ϕA_list[0], color="C0")
#     lineB, = ax.plot(x, ϕB_list[0], color="C1")

#     ax.set(ylim=[-0.1, 1.1])
#     ax.set(ylim=[-0.1, 1.1],
#            xlabel=r"$x$",
#            title=rf"$\alpha = {α}, \delta = {δ}$")
#     ax.grid(axis="y")
#     # ax.legend()
#     ax.text(-30, 0.45, r"$\phi^A$", color="C0", rotation="vertical", va="top")
#     ax.text(-29.5, 0.5, ",", color="k", rotation="vertical")
#     ax.text(-30, 0.55, r"$\phi^B$", color="C3", rotation="vertical", va="bottom")
#     plt.tight_layout()
    
#     def init():
#         return [lineA, lineB]

#     def animate(i):
#         lineA.set_xdata(x) # line plot
#         lineA.set_ydata(ϕA_list[i]) # line plot

#         lineB.set_xdata(x) # line plot
#         lineB.set_ydata(ϕB_list[i]) # line plot

#         return [lineA, lineB]

#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=len(ϕA_list), interval=40, blit=True)

#     HTML(anim.to_html5_video())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder", type=str, default=".",
                        help="full path to folder containing data in .h5 file")
    parser.add_argument("-save", type=bool, default=False,
                        help="whether to save plot or not")
    
    args = parser.parse_args()
    files = sorted(glob(os.path.join(args.datafolder, "*.h5")))
    
    with open(os.path.join(args.datafolder, "params.json"), "r") as p:
        params = json.load(p)
        
    fig, ax = plt.subplots(1, 3, dpi=150, sharex=True, sharey=True, figsize=(9, 4))
    for file in files:
        with h5py.File(file, "r") as d:
            phiA = d['tasks']["phiA"][()]
            phiB = d['tasks']["phiB"][()]
            UA = d['tasks']['UA'][()]
            UB = d['tasks']['UB'][()]
            x = d['tasks']['phiA'].dims[1][0][()]
            t = d['tasks']['phiA'].dims[0]["sim_time"][()]
        a0 = ax[0].pcolormesh(x, t, phiA, cmap="Blues",
                              vmin=0, vmax=1,
                              rasterized=True,
                              shading="gouraud")
        a1 = ax[1].pcolormesh(x, t, phiB, cmap="Reds",
                              vmin=0, vmax=1,
                              rasterized=True,
                              shading="gouraud")
        a2 = ax[2].pcolormesh(x, t, phiA - phiB, cmap="RdBu_r",
                             vmin=-1, vmax=1,
                             rasterized=True,
                             shading="gouraud")

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
    
    if args.save:
        fig.savefig(os.path.join(args.datafolder, "kymo.pdf"))
    plt.tight_layout()
    
    # plot final configuration
    fig, ax = plt.subplots(dpi=150, figsize=(4, 2))
    # initialize line
    ax.plot(x, phiA[-1], color="C0", label=r"$\phi^A$")
    ax.plot(x, phiB[-1], color="C3", label=r"$\phi^B$")
    ax.set(ylim=[-0.1, 1.1],
           xlabel=r"$x$",
           title=rf"$\alpha = {params['alpha']}, \delta = {params['delta']}$")
    ax.grid(axis="y")
    ax.text(-30, 0.45, r"$\phi^A$", color="C0", rotation="vertical", va="top")
    ax.text(-29.5, 0.5, ",", color="k", rotation="vertical")
    ax.text(-30, 0.55, r"$\phi^B$", color="C3", rotation="vertical", va="bottom")
    plt.tight_layout()
    if args.save:
        fig.savefig(os.path.join(args.datafolder, "finalFrame.pdf"))
    
    plt.show()
    