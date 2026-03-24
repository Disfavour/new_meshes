import matplotlib.pyplot as plt
import numpy as np
import utility
import pathlib
import sympy
import sys
# sys.path.append('computations/anisotropic_diffusion_reaction')
# import unsteady_Robin_two_level
#sys.path.append('computations/anisotropic_diffusion_reaction/decoupling')
sys.path.append('computations/anisotropic_diffusion_reaction/unsteady')
import two_level
import two_level_decoupling
import two_level_decoupling2


def plot_err(x, y, legend, fname, ylim=None, ylabel=r'$\varepsilon$'):
    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    ax.plot(x, y, f'o-')

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(ylabel)
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_err_t(x, y, legend, fname, ylim=None, ylabel=r'$\varepsilon$', legend_title=None, base=2):
    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    for xi, yi in zip(x, y):
        ax.plot(xi, yi)

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.grid()

    #ax.set_xscale('log')
    ax.set_yscale('log', base=base)
    ax.legend(legend, title=legend_title)

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_err_bw(xs, ys, legend, fname, ylim=None, ylabel=r'$\varepsilon$', legend_title=None, base=2):
    line_styles = ('-', '--', '-.', ':')

    assert len(ys) <= len(line_styles)

    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    for x, y, ls in zip(xs, ys, line_styles):
        ax.plot(x, y, color='k', linestyle=ls)

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.grid()

    #ax.set_xscale('log')
    ax.set_yscale('log', base=base)
    ax.legend(legend, title=legend_title)

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_err_t_dif_lines(xs, ys, legend, fname, ylim=None, ylabel=r'$\varepsilon$', legend_title=None, base=2):
    cmap = plt.get_cmap('tab10')
    line_styles = ('-', '--', ':', '-.')

    assert len(xs) <= len(line_styles)

    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    for x, y, ls in zip(xs, ys, line_styles):
        for i, xi, yi in enumerate(zip(x, y)):
            ax.plot(xi, yi, color=cmap(i), linestyle=ls)

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.grid()

    #ax.set_xscale('log')
    ax.set_yscale('log', base=base)
    ax.legend(legend, title=legend_title)

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_err_compare(x, ys, legend, fname, ylim=None, ylabel=r'$\varepsilon^D_2$'):
    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    cmap = plt.get_cmap('tab10')
    line_styles = ('-', '--', ':', '-.')
    # line_widths = (3, 2, 1)

    for ls, y in zip(line_styles, ys):
        for i, y in enumerate(y.T):
            ax.plot(x, y, f'{ls}', color=cmap(i))

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(ylabel)
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)

    fig.savefig(fname, transparent=True)
    plt.close()


def setup_problem():
    # div (-k * grad u) + r*u = f
    # q*n - cu = bc
    x, y, nx, ny, t = sympy.symbols('x y nx ny t')
    n = sympy.Matrix([nx, ny])

    #u = sympy.exp(x*y) + 1 / (1 + t)
    
    k = sympy.Matrix([
        [1, 3],
        [3, 10]
    ])
    r = 1
    c = 1

    # x0, y0 = 1, 0.75
    # sx, sy = 1, 0.75
    # u = sympy.exp(-t) * sympy.exp(-sympy.exp(t) * ((x - x0)**2 / 2*sx + (y - y0)**2 / 2*sy))
    # e^{-t}e^{-e^{t}\left(\frac{\left(x-x_{0}\right)^{2}}{2s_{x}}+\frac{\left(y-y_{0}\right)^{2}}{2s_{y}}\right)}

    lambda1 = sympy.pi**2 * (k[0] + 4*k[3]) + r
    
    u = sympy.exp(-(x - 0.2)**2 / (4*k[0]*t + 0.1) - (y - 0.6)**2 / (4*k[3]*t + 0.1))
    #e^{-(x-0.2)^{2}/(4*1*t+0.1)-(y-0.6)^{2}/(4*10*t+0.1)}

    #u = sympy.exp(-lambda1*t) * sympy.sin(sympy.pi*(x + 0.3)) * sympy.sin(2*sympy.pi*(y + 0.2))
    #e^{-\left(1\cdot\pi^{2}+1\right)t}\sin\left(\pi\left(x+0.3\right)\right)\sin\left(2\pi\left(y+0.2\right)\right)

    #u = sympy.exp(-lambda1*t) * (x - 0.5)**2 * (y - 0.3) + 0.1*sympy.sin(3*sympy.pi*x*y)
    #e^{-\left(\pi^{2}\cdot1\ +\ 1\right)t}(x-0.5)^{2}(y-0.3)+0.1\sin(3\pi xy)

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y, t], u.diff(t) + div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny, t], q.dot(n) + c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y, t], u, "numpy")
    q = sympy.lambdify([x, y, t], q, "numpy")    

    return k, r, f, c, bc, u, q


def plot_dif_tau(solve, setup, mesh_number, ts, fname):
    mesh = f'meshes/ellipse/quadrangle_{mesh_number}.msh'
    data = []
    for t in ts:
        data.append(solve(*setup(), t, 1, mesh))

    # nodes.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    plot_err_t(ts, [data[i][1][:, 2] for i in range(len(ts))], [rf'$\tau = {t[1]-t[0]}$' for t in ts], fname, ylabel=r'$\varepsilon_2$')


def plot_dif_meshes(solve, setup, mesh_numbers, t, fname):
    data = []
    for i in mesh_numbers:
        mesh = f'meshes/ellipse/quadrangle_{i}.msh'
        data.append(solve(*setup(), t, mesh)[1])
    data = np.array(data)

    # nodes.shape[0], (L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q)
    plot_err_t([t]*mesh_numbers.size, [d[:, 2] for d in data], [rf'Сетка {i}' for i in mesh_numbers], fname, ylabel=r'$\varepsilon_2$')


def plot_dif(prefix):
    solve = elliptic_mvd_sparse_Robin_unsteady.solve_problem
    setup = setup_problem
    #ts = [np.linspace(0, 1, 2**i + 1) for i in range(6)]
    ts = [np.linspace(0, 1, 10**(i + 2) + 1) for i in range(3)]
    #ts = [np.linspace(0, 1, 5 * 2**i + 1) for i in range(6)]
    meshes = np.arange(5, 7 + 1)
    plot_dif_tau(solve, setup, meshes[-2], ts, f'{prefix}_dif_tau.pdf')
    plot_dif_meshes(solve, setup, meshes, ts[-2], f'{prefix}_dif_meshes.pdf')





def dif_mesh(folder):
    solve = elliptic_mvd_sparse_Robin_unsteady.solve_problem
    setup = setup_problem
    t = np.linspace(0, 1, 100 + 1)
    meshes = np.arange(1, 8)
    plot_dif_meshes(solve, setup, meshes, t, f'{folder}/L2_dif_meshes2.pdf')


def setup_problem1():
    # div (-k * grad u) + r*u = f
    # q*n - cu = bc
    x, y, nx, ny, t = sympy.symbols('x y nx ny t')
    n = sympy.Matrix([nx, ny])

    k = np.array((
        (1, 3),
        (3, 10)
    ))

    # x0, y0 = 1, 0.75
    # sx, sy = 1, 0.75
    # u = sympy.exp(-t) * sympy.exp(-sympy.exp(t) * ((x - x0)**2 / 2*sx + (y - y0)**2 / 2*sy))
    
    #u = sympy.exp(-(x - 0.2)**2 / (4*k[0, 0]*t + 0.1) - (y - 0.6)**2 / (4*k[1, 1]*t + 0.1))

    # e^{-\left(1+10tx\right)y^{2}}
    u = sympy.exp(-(1 + 10*t*x)*y**2)
    
    r = 1
    c = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f_f = sympy.lambdify([x, y, t], u.diff(t) + div + r*u, "numpy")
    bc_f = sympy.lambdify([x, y, nx, ny, t], -q.dot(n) + c*u, "numpy")

    u_f = sympy.lambdify([x, y, t], u, "numpy")
    q_f = sympy.lambdify([x, y, t], q, "numpy")

    return k, r, f_f, c, bc_f, u_f, q_f

def dif_tau(folder):
    solve = unsteady_Robin_two_level.solve
    setup = setup_problem1
    ts = [np.linspace(0, 1, 10**(i + 2) + 1) for i in range(3)]
    meshes = np.arange(1, 7)
    for mesh in meshes:
        plot_dif_tau(solve, setup, mesh, ts, f'{folder}/L2_dif_tau3_mesh{mesh}.pdf')


def collect_data():
    solve = unsteady_Robin_two_level.solve
    setup = setup_problem1
    ts = [np.linspace(0, 1, 10**(i + 2) + 1) for i in range(3)]
    meshes = np.arange(1, 7)
    sigmas = [0.5]#[0.5, 1]
    for m in meshes:
        mesh_fname = f'meshes/ellipse/quadrangle_{m}.msh'
        for i, t in enumerate(ts, 1):
            for sigma in sigmas:
                nodes, norms = unsteady_Robin_two_level.solve(*setup_problem1(), t, sigma, mesh_fname)
                np.savez_compressed(f'data/anisotropic_diffusion_reaction/unsteady/m{m}_t{i}_sigma{sigma}.npz', norms=norms, t=t)




def plot_dif_tau1(folder):
    meshes = np.arange(1, 7)
    ts = np.arange(1, 4)
    sigmas = [0.5, 1]
    t_legend = [rf'$\tau = 10^{{{-i}}}$' for i in range(2, 5)]
    for mesh in meshes:
        for sigma in sigmas:
            norms = []
            tsteps = []
            for t in ts:
                loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/m{mesh}_t{t}_sigma{sigma}.npz')
                norms.append(loaded['norms'])
                tsteps.append(loaded['t'])
            plot_err_t(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_dif2_tau_mesh{mesh}_sigma{sigma}.pdf', ylabel=r'$\varepsilon_2$')


def plot_dif_mesh1(folder):
    meshes = np.arange(1, 7)
    ts = np.arange(1, 4)
    sigmas = [0.5, 1]
    mesh_legend = [rf'Сетка ${mesh}$' for mesh in meshes]
    for t in ts:
        for sigma in sigmas:
            norms = []
            for mesh in meshes:
                loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/m{mesh}_t{t}_sigma{sigma}.npz')
                norms.append(loaded['norms'])
            tsteps = loaded['t']
            plot_err_t([tsteps]*len(meshes), [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_dif2_mesh_t{t}_sigma{sigma}.pdf', ylabel=r'$\varepsilon_2$', base=4)


def collect_data2():
    tn = [2**i for i in range(1, 11)]
    ts = [np.linspace(0, 1, i + 1) for i in tn]
    meshes = np.arange(1, 8)#np.arange(1, 10)
    sigmas = [0.5, 1]
    for m in meshes:
        mesh_fname = f'meshes/ellipse/quadrangle_{m}.msh'
        for i, t in zip(tn, ts):
            for sigma in sigmas:
                fname = f'data/anisotropic_diffusion_reaction/unsteady/sigma{sigma}_m{m}_t{i}.npz'
                if pathlib.Path(fname).is_file():
                    continue
                nodes, norms = two_level.solve(*setup_problem1(), t, sigma, mesh_fname)
                np.savez_compressed(fname, norms=norms, t=t)


def plot_dif_tau2(folder, meshes, tn, tnn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady'):
    #t_legend = [rf'$t_n = {i}$' for i in range(2, 5)]
    t_legend = [rf'$2^{{{-i}}}$' for i in tnn]
    for mesh in meshes:
        for sigma in sigmas:
            norms = []
            tsteps = []
            for t in tn:
                loaded = np.load(f'{prefix}/sigma{sigma}_m{mesh}_t{t}.npz')
                norms.append(loaded['norms'])
                tsteps.append(loaded['t'])
            #t_legend = [rf'$\tau = {i[1] - i[0]}$' for i in tsteps]
            #t_legend = [rf'${i[1] - i[0]}$' for i in tsteps]
            plot_err_t(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_dif_tau_sigma{sigma}_mesh{mesh}.pdf', ylabel=r'$\varepsilon_2$', legend_title=rf'$\tau$')


def plot_dif_mesh2(folder, meshes, tn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady'):
    mesh_legend = [rf'${mesh}$' for mesh in meshes]
    for t in tn:
        for sigma in sigmas:
            norms = []
            for mesh in meshes:
                loaded = np.load(f'{prefix}/sigma{sigma}_m{mesh}_t{t}.npz')
                norms.append(loaded['norms'])
            tsteps = loaded['t']
            plot_err_t([tsteps]*len(meshes), [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_dif_mesh_sigma{sigma}_t{t}.pdf', ylabel=r'$\varepsilon_2$', legend_title=rf'$N$', base=4)


def plot_all(folder):
    meshes = np.arange(1, 8)
    tnn = [i for i in range(1, 11)]
    tn = [2**i for i in tnn]
    sigmas = [0.5, 1]
    for sigma in sigmas:
        plot_dif_tau2(folder, meshes, tn, tnn, [sigma])
    plot_dif_mesh2(folder, meshes, tn, sigmas)

    # [100 * 2**i for i in range(5)]
    # [10 * 2**i for i in range(5)]
    # [2**i for i in range(6)]
    # [10 * 2**i for i in range(8)]

    # tn = [2**i for i in range(2, 10)] mesh 1-7 ok
    # tn = [2**i for i in range(2, 8)] mesh 

    # сетка 8 7 сек/итерация
    # сетка 8 70 сек/итерация



def collect_data_decoupling():
    time_steps_numbers = [2**i for i in range(1, 13)]
    time_steps = [np.linspace(0, 1, i + 1) for i in time_steps_numbers]
    meshes = np.arange(1, 7)#np.arange(1, 10)
    sigmas = [0.5, 0.75, 1, 1.25]
    for m in meshes:
        mesh_fname = f'meshes/ellipse/quadrangle_{m}.msh'
        for time_steps_number, ts in zip(time_steps_numbers, time_steps):
            for sigma in sigmas:
                fname = f'data/anisotropic_diffusion_reaction/unsteady_decoupling/sigma{sigma}_m{m}_t{time_steps_number}.npz'
                if pathlib.Path(fname).is_file():
                    continue
                nodes, norms = two_level_decoupling.solve(*setup_problem1(), ts, sigma, mesh_fname)
                np.savez_compressed(fname, norms=norms, t=ts)


def plot_decoupling(folder):
    meshes = np.arange(1, 7)
    tnn = [i for i in range(3, 13)]
    tn = [2**i for i in tnn]
    sigmas = [0.5, 0.75, 1, 1.25]
    for sigma in sigmas:
        plot_dif_tau2(folder, meshes, tn, tnn, [sigma], prefix='data/anisotropic_diffusion_reaction/unsteady_decoupling')
    plot_dif_mesh2(folder, meshes, tn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady_decoupling')


def collect_data_decoupling2():
    time_steps_numbers = [2**i for i in range(1, 15)]
    time_steps = [np.linspace(0, 1, i + 1) for i in time_steps_numbers]
    meshes = np.arange(1, 8)
    sigmas = [1]#[0.5, 0.75, 1, 1.25]
    for m in meshes:
        mesh_fname = f'meshes/ellipse/quadrangle_{m}.msh'
        for time_steps_number, ts in zip(time_steps_numbers, time_steps):
            for sigma in sigmas:
                #path = f'data/anisotropic_diffusion_reaction/unsteady/two_level_decoupling2_sigma{sigma}_m{m}_t{time_steps_number}.npz'
                path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'unsteady' / f'two_level_decoupling2_sigma{sigma}_m{m}_t{time_steps_number}.npz'
                if path.is_file():
                    continue
                nodes, norms = two_level_decoupling2.solve(*setup_problem1(), ts, sigma, mesh_fname)
                np.savez_compressed(path, norms=norms, t=ts)


def plot_dif_tau3(folder, meshes, tn, tnn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady'):
    #t_legend = [rf'$t_n = {i}$' for i in range(2, 5)]
    t_legend = [rf'$2^{{{-i}}}$' for i in tnn]
    for mesh in meshes:
        for sigma in sigmas:
            norms = []
            tsteps = []
            for t in tn:
                loaded = np.load(f'{prefix}/two_level_decoupling2_sigma{sigma}_m{mesh}_t{t}.npz')
                norms.append(loaded['norms'])
                tsteps.append(loaded['t'])
            #t_legend = [rf'$\tau = {i[1] - i[0]}$' for i in tsteps]
            #t_legend = [rf'${i[1] - i[0]}$' for i in tsteps]
            plot_err_t(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_dif_tau_sigma{sigma}_mesh{mesh}.pdf', ylabel=r'$\varepsilon_2$', legend_title=rf'$\tau$')


def plot_dif_mesh3(folder, meshes, tn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady'):
    mesh_legend = [rf'${mesh}$' for mesh in meshes]
    for t in tn:
        for sigma in sigmas:
            norms = []
            for mesh in meshes:
                loaded = np.load(f'{prefix}/two_level_decoupling2_sigma{sigma}_m{mesh}_t{t}.npz')
                norms.append(loaded['norms'])
            tsteps = loaded['t']
            plot_err_t([tsteps]*len(meshes), [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_dif_mesh_sigma{sigma}_t{t}.pdf', ylabel=r'$\varepsilon_2$', legend_title=rf'$N$', base=4)


def plot_decoupling2(folder):
    meshes = np.arange(1, 8)
    tnn = [i for i in range(5, 15)]
    tn = [2**i for i in tnn]
    sigmas = [1]#[0.5, 0.75, 1, 1.25]
    for sigma in sigmas:
        plot_dif_tau3(folder, meshes, tn, tnn, [sigma], prefix='data/anisotropic_diffusion_reaction/unsteady')
    plot_dif_mesh3(folder, meshes, tn, sigmas, prefix='data/anisotropic_diffusion_reaction/unsteady')


def plot_two_level_dif_tau_bw(folder):
    mesh = 6
    tn = [i for i in range(7, 11)]
    ts = [2**i for i in tn]
    sigma = 1
    t_legend = [rf'$2^{{{-i}}}$' for i in tn]

    norms = []
    tsteps = []
    for t in ts:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/sigma{sigma}_m{mesh}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_dif_tau_sigma{sigma}_mesh{mesh}.pdf', legend_title=rf'$\tau$')


def plot_two_level_dif_mesh_bw(folder):
    meshes = np.arange(4, 8)
    t = 2**9
    sigma = 1
    mesh_legend = [rf'{mesh}' for mesh in range(1, len(meshes) + 1)]
    norms = []
    tsteps = []
    for m in meshes:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/sigma{sigma}_m{m}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_dif_mesh_sigma{sigma}_tau{t}.pdf', legend_title=rf'$N$')


def plot_two_level_dif_tau_bw2(folder):
    mesh = 6
    tn = [i for i in range(1, 5)]
    ts = [2**i for i in tn]
    sigma = 0.5
    t_legend = [rf'$2^{{{-i}}}$' for i in tn]

    norms = []
    tsteps = []
    for t in ts:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/sigma{sigma}_m{mesh}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_dif_tau_sigma{sigma}_mesh{mesh}.pdf', legend_title=rf'$\tau$')


def plot_two_level_dif_mesh_bw2(folder):
    meshes = np.arange(4, 8)
    t = 2**3
    sigma = 0.5
    mesh_legend = [rf'{mesh}' for mesh in range(1, len(meshes) + 1)]
    norms = []
    tsteps = []
    for m in meshes:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/sigma{sigma}_m{m}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_dif_mesh_sigma{sigma}_tau{t}.pdf', legend_title=rf'$N$')


def plot_two_level_dif_tau_bw_decoupling(folder):
    mesh = 6
    tn = [i for i in range(11, 15)]
    ts = [2**i for i in tn]
    sigma = 1
    t_legend = [rf'$2^{{{-i}}}$' for i in tn]

    norms = []
    tsteps = []
    for t in ts:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/two_level_decoupling2_sigma{sigma}_m{mesh}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_decoupling_dif_tau_sigma{sigma}_mesh{mesh}.pdf', legend_title=rf'$\tau$')


def plot_two_level_dif_mesh_bw_decoupling(folder):
    meshes = np.arange(4, 8)
    t = 2**13
    sigma = 1
    mesh_legend = [rf'{mesh}' for mesh in range(1, len(meshes) + 1)]
    norms = []
    tsteps = []
    for m in meshes:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/two_level_decoupling2_sigma{sigma}_m{m}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_decoupling_dif_mesh_sigma{sigma}_tau{t}.pdf', legend_title=rf'$N$')


def plot_two_level_dif_tau_bw_decoupling2(folder):
    mesh = 6
    tn = [i for i in range(11, 15)]
    ts = [2**i for i in tn]
    sigma = 0.5
    t_legend = [rf'$2^{{{-i}}}$' for i in tn]

    norms = []
    tsteps = []
    for t in ts:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/two_level_decoupling2_sigma{sigma}_m{mesh}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], t_legend, f'{folder}/L2_decoupling_dif_tau_sigma{sigma}_mesh{mesh}.pdf', legend_title=rf'$\tau$')


def plot_two_level_dif_mesh_bw_decoupling2(folder):
    meshes = np.arange(4, 8)
    t = 2**13
    sigma = 0.5
    mesh_legend = [rf'{mesh}' for mesh in range(1, len(meshes) + 1)]
    norms = []
    tsteps = []
    for m in meshes:
        loaded = np.load(f'data/anisotropic_diffusion_reaction/unsteady/two_level_decoupling2_sigma{sigma}_m{m}_t{t}.npz')
        norms.append(loaded['norms'])
        tsteps.append(loaded['t'])

    plot_err_bw(tsteps, [norms[i][:, 2] for i in range(len(norms))], mesh_legend, f'{folder}/L2_decoupling_dif_mesh_sigma{sigma}_tau{t}.pdf', legend_title=rf'$N$')


if __name__ == '__main__':
    folder = 'images/mvd/unsteady'
    #dif_tau(folder)
    #collect_data()
    #plot_dif_tau1(folder)
    #plot_dif_mesh1(folder)

    folder = 'images/unsteady_anisotropic_diffusion_reaction/errors'
    #collect_data2()
    #plot_all(folder)

    folder = 'images/unsteady_anisotropic_diffusion_reaction/decoupling'
    #collect_data_decoupling()
    #plot_decoupling(folder)

    folder = 'images/unsteady_anisotropic_diffusion_reaction/decoupling2'
    #collect_data_decoupling2()
    plot_decoupling2(folder)

    folder = 'images/unsteady_anisotropic_diffusion_reaction/errors_bw'
    # plot_two_level_dif_tau_bw(folder)
    # plot_two_level_dif_mesh_bw(folder)
    # plot_two_level_dif_tau_bw2(folder)
    # plot_two_level_dif_mesh_bw2(folder)
    # plot_two_level_dif_tau_bw_decoupling(folder)
    # plot_two_level_dif_mesh_bw_decoupling(folder)
    # plot_two_level_dif_tau_bw_decoupling(folder)
    # plot_two_level_dif_mesh_bw_decoupling(folder)

    #plot_dif('images/mvd/unsteady/L2_log3')

    #tau = 1e-4 ,1e-3, 1e-2
    # + картинки решения
    # + выбрали 1ю функцию

    # import gmsh
    # gmsh.initialize()
    # gmsh.open(f'meshes/ellipse/quadrangle_{5}.msh')

    # quadrangle_tags, quadrangles = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    # tmp = gmsh.model.mesh.get_element_qualities(quadrangle_tags, qualityName='maxEdge')
    # print(tmp)
    # print(tmp.shape)
    # print(tmp.mean(), tmp.min(), tmp.max())

    # gmsh.fltk.run()

    #gmsh.close()
