import matplotlib.pyplot as plt
import numpy as np
import sys
from multiprocessing import Pool
import utility
import gmsh
import matplotlib.collections
from matplotlib import collections, cm, colors
import matplotlib.tri
import sympy
from matplotlib.colors import Normalize
sys.path.append('computations/mvd_iterative_method')
import two_level
import two_level_minimum_correction_method
import three_level
import gamma_for_cells
import direct_monolitic
import direct_fem


def plot_r(errs, legend, fname, ylabel=r'$r$'):
    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)

    assert errs.shape[0] <= 20
    cmap = plt.get_cmap('tab10') if errs.shape[0] <= 10 else plt.get_cmap('tab20')
    ncols = 1 if errs.shape[0] <= 10 else 2

    for i, err in enumerate(errs):
        ax.plot(err, color=cmap(i))
    
    ax.set_xlabel("$k$")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(legend, ncols=ncols)

    #ax.set_xscale('log')
    ax.set_yscale('log')

    #yticks = ax.get_yticks()
    #print(yticks)
    #ax.set_ylim(yticks[1], yticks[-2])

    fig.savefig(fname, transparent=True)

    plt.close()


def compare_3_methods():
    mesh_fname = 'meshes/rectangle/rectangle_9_quadrangle'
    max_iter = 1000

    sigma = 10
    kappa = 100
    k = np.array(
        ((1, sigma),
        (sigma, kappa))
    )

    with Pool(6) as p:
        res1 = p.apply_async(two_level.calculate, (mesh_fname, k, max_iter))
        res2 = p.apply_async(two_level_minimum_correction_method.calculate, (mesh_fname, k, max_iter))
        res3 = p.apply_async(three_level.calculate, (mesh_fname, k, max_iter))
        res1.wait()
        res2.wait()
        res3.wait()
    
    data = np.stack((res1.get(), res2.get(), res3.get()))

    legend = ['two_level', 'two_level_minimum_correction_method', 'three_level']
    plot_r(data, legend, f'images/mvd_iterational_method/compare_3_methods_kappa{kappa}_s{sigma}.pdf', r'$||r^k||_2$')


def latex_table():
    meshes = [11, 15, 18] # [1, 4, 9] [11, 15, 18]
    kappas = [1, 10, 100]
    sigmas = ((0.5, 0.95), (0, 1.5, 3), (0, 5, 9.5))

    kappa_sigma = [(kappa, sigma) for kappa, ss in zip(kappas, sigmas) for sigma in ss]

    ks = [
        np.array(
            ((1, sigma),
            (sigma, kappa))
        ) for kappa, sigma in kappa_sigma
    ]

    mesh_fnames = [f'meshes/rectangle/rectangle_{mesh}_quadrangle' for mesh in meshes]
    
    max_iter = 10000
    r_rel_min = 1e-6
    tau = 1

    # quadrangle_mesh_name, k, max_iter, tau=None, r_rel_min=-1
    data = []
    for i in ((mesh_fname, k, max_iter, tau, r_rel_min) for mesh_fname in mesh_fnames for k in ks):
        r_abs, r_rel2, gamma2, tau = two_level.calculate(*i, info=True)
        r_abs, r_rel3, gamma3, tau = three_level.calculate(*i, info=True)

        assert gamma2 == gamma3

        data.append((gamma2, r_rel2.size - 1, r_rel3.size - 1))
        print(data[-1])
    
    assert len(data) == 24
    data_mesh1 = data[:8]
    data_mesh2 = data[8:16]
    data_mesh3 = data[16:]

    for (kappa, sigma), d1, d2, d3 in zip(kappa_sigma, data_mesh1, data_mesh2, data_mesh3):
        # iterations = [f'{d[1]}/{d[2]}' if d[1] != d[2] else f'{d[1]}' for d in (d1, d2, d3)]
        # row = [f'{kappa}', f'{sigma:.1f}', f'{d1[0]:.6f}', , f'{d2[0]:.6f}', f'{d3[0]:.6f}']

        list_of_lists = [(f'{d[0]:.8f}', f'{d[1]}/{d[2]}' if d[1] != d[2] else f'{d[1]}') for d in (d1, d2, d3)]
        flatten_list = [i for l in list_of_lists for i in l]

        row = [f'{kappa}', f'{sigma:.2f}'] + flatten_list

        print(fr'{' & '.join(row)} \\ \hline')


def local_gammas():
    mesh = 11
    mesh_fname = f'meshes/rectangle/rectangle_{mesh}_quadrangle'
    sigma = 1.5
    kappa = 10
    k = np.array(
        ((1, sigma),
        (sigma, kappa))
    )

    quads, jl = gamma_for_cells.calculate(mesh_fname, k)
    jl = np.abs(jl)
    print(fr'${jl.min():.8f} \le \gamma_l \le {jl.max():.8f}$')

    pc = matplotlib.collections.PolyCollection(quads, edgecolors='none')
    pc.set_array(jl)
    pc.set_norm(Normalize(vmin=0, vmax=1))

    fig, ax = plt.subplots(figsize=utility.get_figsize(1 + 0.15, 0.75), constrained_layout=True)
    ax.add_collection(pc)

    fig.colorbar(pc)

    ax.margins(x=0, y=0)
    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(f'images/mvd_iterational_method/local_gammas_mesh{mesh}_kappa{kappa}_sigma{sigma}.pdf', transparent=True)
    plt.close()


def plot_solution():
    mesh = 18 #18
    mesh_fname = f'meshes/rectangle/rectangle_{mesh}_quadrangle'
    kappa = 10
    sigmas = [0, 3]

    params = [(kappa, sigma) for sigma in sigmas]

    ks = [
        np.array(
            ((1, sigma),
            (sigma, kappa))
        ) for kappa, sigma in params
    ]

    data = []
    for i in ((mesh_fname, k) for k in ks):
        data.append(direct_monolitic.calculate(*i))

    node_coords, u = data[0]
    x, y = node_coords.T
    triangulation = matplotlib.tri.Triangulation(x, y)

    for (kappa, sigma), d in zip(params, data):
        node_coords, u = d

        fig, ax = plt.subplots(figsize=utility.get_figsize(1 + 0.15, 0.75), constrained_layout=True)
        
        lines = ax.tricontourf(triangulation, u)
        ax.tricontour(triangulation, u, colors='k', linewidths=0.5)

        fig.colorbar(lines)
        
        ax.axis('scaled')
        ax.set_axis_off()

        fig.savefig(f'images/mvd_iterational_method/solution_mesh{mesh}_kappa{kappa}_sigma{sigma}.pdf', transparent=True)
        plt.close()


def different_taus_for_two_level():
    mesh = 15 # 15
    mesh_fname = f'meshes/rectangle/rectangle_{mesh}_quadrangle'
    kappa = 10
    sigma = 1.5
    
    k = np.array(
        ((1, sigma),
        (sigma, kappa))
    )

    taus = (0.8, 1, 1.05, 1.1)

    max_iter = 1000

    data1 = two_level.calculate(mesh_fname, k, max_iter, info=True)
    
    data2 = []
    for i in ((mesh_fname, k, max_iter, tau) for tau in taus):
        data2.append(two_level.calculate(*i, info=True))

    tau_critical = data1[3]
    print(fr'$\tau^* = {tau_critical:.8f}$')

    data2.append(data1)
    data2.sort(key=lambda x: x[3], reverse=True)

    data = np.array([i[1] for i in data2])
    taus = [i[3] for i in data2]

    legend = [fr'$\tau = {tau}$' if tau != tau_critical else r'$\tau = \tau^*$' for tau in taus]
    plot_r(data, legend, f'images/mvd_iterational_method/two_level_different_taus_mesh{mesh}_kappa{kappa}_sigma{sigma}.pdf', r'$\varepsilon^k$')


def compare_two_and_three_level():
    mesh = 15 # 15
    mesh_fname = f'meshes/rectangle/rectangle_{mesh}_quadrangle'
    kappa = 10
    sigmas = (0, 1.5, 3)

    params = [(kappa, sigma) for sigma in sigmas]

    ks = [
        np.array(
            ((1, sigma),
            (sigma, kappa))
        ) for kappa, sigma in params
    ]

    max_iter = 1000

    data1, data2 = [], []
    for i in ((mesh_fname, k, max_iter) for k in ks):
        data1.append(two_level.calculate(*i, info=True))
        data2.append(three_level.calculate(*i, info=True))

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]

    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)

    cmap = plt.get_cmap('tab10')
    for i, (d1, d2, label) in enumerate(zip(data1, data2, legend)):
        ax.plot(d1[1], '--',  color=cmap(i))
        ax.plot(d2[1], '-', color=cmap(i), label=label)
    
    ax.set_xlabel("$k$")
    ax.set_ylabel(r'$\varepsilon^k$')
    ax.grid()
    ax.legend()

    ax.set_yscale('log')

    ax.set_ylim(top=100)

    fig.savefig(f'images/mvd_iterational_method/compare_two_and_three_level_mesh{mesh}_kappa{kappa}.pdf', transparent=True)

    plt.close()


if __name__ == '__main__':
    #local_gammas()
    #different_taus_for_two_level()
    #compare_two_and_three_level()
    #plot_solution()
    latex_table()