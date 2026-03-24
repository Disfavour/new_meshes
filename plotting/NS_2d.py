import matplotlib.pyplot as plt
import numpy as np
import sympy
import matplotlib.tri as tri
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import matplotlib.ticker as ticker
import utility
import meshio
import os.path


def get_quantile_levels_positive(u, n=5):
    return np.quantile(u[u > 0], np.linspace(0, 1, n + 1)[1:])

def get_quantile_levels_negative(u, n=5, max_value=0):
    return np.quantile(u[u < max_value], np.linspace(0, 1, n + 1)[:-1])

def get_quantile_levels(u, n=5):
    return np.concatenate((get_quantile_levels_negative(u, n), [0], get_quantile_levels_positive(u, n)))

def fmt(x, pos):
    if x == 0:
        return r'$0$'
    a, b = f'{x:.7e}'.split('e')
    b = int(b)
    if b == 0:
        return fr'${{{a}}}$'
    return fr'${a} \cdot 10^{{{b}}}$'


def fmt2(x, pos):
    return fr'${sympy.latex(sympy.N(x, 3))}$'


# extend=both
def plot1(u, coords, cells, fname, cmap_name = 'jet'):
    triangulation = tri.Triangulation(*coords[:, :2].T, cells)

    u_levels = get_quantile_levels(u)[1:-1]
    cmap = cm.get_cmap(cmap_name)
    norm = BoundaryNorm(u_levels, ncolors=cmap.N, extend='both')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1))

    lines = ax.tricontourf(triangulation, u, u_levels, cmap=cmap_name, norm=norm, extend='both')
    ax.tricontour(triangulation, u, u_levels, colors='k', linestyles='-', linewidths=0.5)
    fig.colorbar(lines, ticks=u_levels, format=ticker.FuncFormatter(fmt))

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


def plot(u, triangulation, levels, fname, fmt=fmt, cmap_name='turbo'):
    cmap = cm.get_cmap(cmap_name)
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    # utility.get_figsize_2_columns(1.45, 1)
    fig, ax = plt.subplots(figsize=utility.get_figsize(1.45, 1))

    lines = ax.tricontourf(triangulation, u, levels, cmap=cmap_name, norm=norm)
    ax.tricontour(triangulation, u, levels, colors='k', linestyles='-', linewidths=0.5)
    fig.colorbar(lines, ticks=levels, format=ticker.FuncFormatter(fmt))

    ax.axis('scaled')
    #ax.set_axis_off()
    ax.tick_params(
        which='both',      # both major and minor ticks
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,        # ticks along the left edge are off
        right=False,       # ticks along the right edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False    # labels along the left edge are off
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)
    plt.close()


def plot_ellipse(u, triangulation, levels, fname, fmt=fmt, cmap_name='turbo'):
    cmap = cm.get_cmap(cmap_name)
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), layout="constrained")

    lines = ax.tricontourf(triangulation, u, levels, cmap=cmap_name, norm=norm)
    ax.tricontour(triangulation, u, levels, colors='k', linestyles='-', linewidths=0.5)
    cbar = fig.colorbar(lines, ax=ax, ticks=levels, format=ticker.FuncFormatter(fmt), orientation='vertical')
    #cbar.ax.tick_params(axis='x', rotation=270)
    # vertical horizontal 

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def read_npz(fname):
    data = np.load(fname)
    dofs_coords = data['dofs_coords']
    psi_w = data['psi_w']
    uv = data['uv']
    
    psi = psi_w[::2]
    w = psi_w[1::2]
    u = uv[::2]
    v = uv[1::2]
    return dofs_coords, psi, w, u, v


def read_xdmf(fname):
    with meshio.xdmf.TimeSeriesReader(fname) as reader:
        points, cells = reader.read_points_cells()
        t, point_data, cell_data = reader.read_data(0)
    
    x, y = points[:, 0], points[:, 1]
    triangulation = tri.Triangulation(x, y, cells[0].data)
    psi, w, u, v = point_data['f'].T
    return triangulation, psi, w, u, v


def all_npz():
    m, p = 800, 2
    dofs_coords, psi, w, u, v = read_npz(f'data/NS/stokes_n{m}_p{p}.npz')
    x, y = dofs_coords[:, 0], dofs_coords[:, 1]
    triangulation = tri.Triangulation(x, y)
    
    pref = f'images/NS/stokes_m{m}_p{p}'
    for f, f_name in  zip((psi, w, u, v), ('psi', 'w', 'u', 'v')):
        levels = get_quantile_levels(f)
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')
    
    Re = 1000
    dofs_coords, psi, w, u, v = read_npz(f'data/NS/Re{Re}_n{m}_p{p}.npz')
    pref = f'images/NS/Re{Re}_m{m}_p{p}'
    for f, f_name in  zip((psi, w, u, v), ('psi', 'w', 'u', 'v')):
        levels = get_quantile_levels(f)
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')
    
    Re = 10000
    dofs_coords, psi, w, u, v = read_npz(f'data/NS/Re{Re}_n{m}_p{p}.npz')
    pref = f'images/NS/Re{Re}_m{m}_p{p}'
    for f, f_name in  zip((w, u, v), ('w', 'u', 'v')):
        levels = get_quantile_levels(f)
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')
    
    # custom level for psi (right bot) -> quantile 99
    #custom_level = -1e-5
    levels_positive = get_quantile_levels_positive(psi)
    #levels_negative = get_quantile_levels_negative(psi, 4, custom_level)
    q_negative = np.linspace(0, 1, (5 - 1) + 1)[:-1]
    q_negative = np.concatenate((q_negative, [0.99]))
    levels_negative =  np.quantile(psi[psi < 0], q_negative)
    levels = np.concatenate((levels_negative, [0], levels_positive))
    plot(psi, triangulation, levels, f'{pref}_psi.pdf')

def stokes():
    Re = 0
    m = 800
    p = 2
    triangulation, psi, w, u, v = read_xdmf(f'data/NS/Re{Re}_n{m}_p{p}.xdmf')
    extreme_values = np.load(f'data/NS/extreme_values_Re{Re}_n{m}_p{p}.npz')['extreme_values'].reshape(-1, 2)

    pref = f'images/NS/stokes_m{m}_p{p}'

    for f, f_name, (fmin, fmax) in zip((psi, w, u, v), ('psi', 'w', 'u', 'v'), extreme_values):
        levels = get_quantile_levels(f)
        levels[0] = fmin
        levels[-1] = fmax
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')


def Re_1000():
    Re = 1000
    m = 800
    p = 2
    triangulation, psi, w, u, v = read_xdmf(f'data/NS/Re{Re}_n{m}_p{p}.xdmf')
    extreme_values = np.load(f'data/NS/extreme_values_Re{Re}_n{m}_p{p}.npz')['extreme_values'].reshape(-1, 2)

    pref = f'images/NS/Re{Re}_m{m}_p{p}'

    for f, f_name, (fmin, fmax) in zip((psi, w, u, v), ('psi', 'w', 'u', 'v'), extreme_values):
        levels = get_quantile_levels(f)
        levels[0] = fmin
        levels[-1] = fmax
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')


def Re_10000():
    Re = 10000
    m = 800
    p = 2
    triangulation, psi, w, u, v = read_xdmf(f'data/NS/Re{Re}_n{m}_p{p}.xdmf')
    extreme_values = np.load(f'data/NS/extreme_values_Re{Re}_n{m}_p{p}.npz')['extreme_values'].reshape(-1, 2)

    pref = f'images/NS/Re{Re}_m{m}_p{p}'

    custom_level = -1e-5
    psi_min, psi_max = extreme_values[0]
    levels_positive = get_quantile_levels_positive(psi)
    levels_positive[-1] = psi_max
    levels_negative = get_quantile_levels_negative(psi, 4, custom_level)
    levels_negative[0] = psi_min
    levels = np.concatenate((levels_negative, (custom_level, 0), levels_positive))
    plot(psi, triangulation, levels, f'{pref}_psi.pdf')

    for f, f_name, (fmin, fmax) in zip((w, u, v), ('w', 'u', 'v'), extreme_values[1:]):
        levels = get_quantile_levels(f)
        levels[0] = fmin
        levels[-1] = fmax
        plot(f, triangulation, levels, f'{pref}_{f_name}.pdf')

def ellipse():
    p = 2
    Res = [5500, 10000]
    sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    methods = (3, 4)

    pref = 'images/NS/ellipse'
    for Re in Res:
        for method in methods:
            for sigma in sigmas:
                xdmf_fname = f'data/NS/ellipse_128_method_{method}_Re{Re}_s{sigma}_p{p}.xdmf'
                npz_fname = f'data/NS/ellipse_128_method_{method}_Re{Re}_s{sigma}_p{p}.npz'

                if not (os.path.exists(xdmf_fname) and os.path.exists(npz_fname)):
                    continue
                
                triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
                extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

                for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
                    levels = get_quantile_levels(f)
                    levels[0] = fmin
                    levels[-1] = fmax
                    plot_ellipse(f, triangulation, levels, f'{pref}_Re{Re}_method{method}_s{sigma}_{f_name}.pdf')


def ellipse_256():
    Re = 10000
    kappa = 0.5
    sigmas1 = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    sigmas2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'
    prefix_images = f'images/NS/ellipse_256_Re{Re}'

    for sigma in sigmas1:
        xdmf_fname = f'{prefix}_k{kappa}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_k{kappa}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_k{kappa}_s{sigma}_{f_name}.pdf')
    
    for sigma in sigmas2:
        xdmf_fname = f'{prefix}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_s{sigma}_{f_name}.pdf')


def ellipse_256_2():
    Re = 10000
    kappa = 0.5
    sigmas = [0.116, 0.117, 0.118] + [0.119, 0.12, 0.121]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'
    prefix_images = f'images/NS/ellipse_256_Re{Re}'

    for sigma in sigmas:
        xdmf_fname = f'{prefix}_k{kappa}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_k{kappa}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_k{kappa}_s{sigma}_{f_name}.pdf')


def ellipse_256_3():
    Re = 10000
    kappa = 0.5
    sigmas = [0.1155, 0.1165, 0.1175, 0.1185, 0.1195, 0.1205]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'
    prefix_images = f'images/NS/ellipse_256_Re{Re}'

    for sigma in sigmas:
        xdmf_fname = f'{prefix}_k{kappa}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_k{kappa}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_k{kappa}_s{sigma}_{f_name}.pdf')


def ellipse_256_4():
    Re = 10000
    kappa = 0.5
    sigmas = [0.1165, 0.117, 0.1175, 0.118, 0.1185, 0.119, 0.1195, 0.12, 0.1205]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_3_Re{Re}'
    prefix_images = f'images/NS/ellipse_3_Re{Re}'

    for sigma in sigmas:
        xdmf_fname = f'{prefix}_k{kappa}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_k{kappa}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi,), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_k{kappa}_s{sigma}_{f_name}.pdf')


def ellipse_256_5():
    Re = 10000
    kappa = 0.5
    sigmas = [0.117, 0.12]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_3_Re{Re}'
    prefix_images = f'images/NS/ellipse_3_Re{Re}'

    for sigma in sigmas:
        xdmf_fname = f'{prefix}_k{kappa}_s{sigma}.xdmf'
        npz_fname = f'{prefix}_k{kappa}_s{sigma}.npz'
        
        triangulation, psi, w, u, v = read_xdmf(xdmf_fname)
        extreme_values = np.load(npz_fname)['extreme_values'].reshape(-1, 2)

        for f, f_name, (fmin, fmax) in zip((psi, w, u, v), ('psi', 'w', 'u', 'v'), extreme_values):
            levels = get_quantile_levels(f)
            levels[0] = fmin
            levels[-1] = fmax
            plot_ellipse(f, triangulation, levels, f'{prefix_images}_k{kappa}_s{sigma}_{f_name}.pdf')


if __name__ == '__main__':
    # stokes()
    # Re_1000()
    # Re_10000()

    #all_npz()

    # ellipse()
    # ellipse_256()
    # ellipse_256_2()
    #ellipse_256_3()
    # ellipse_256_4()
    ellipse_256_5()