import matplotlib.pyplot as plt
import matplotlib.tri as tri
#from matplotlib.tri import Triangulation, LinearTriInterpolator
import numpy as np
import utility
import pathlib
import sympy
import gmsh
#from mvd_errors_unsteady import plot_err_bw

import sys
sys.path.append('computations/anisotropic_diffusion_reaction/unsteady_article')
import two_level_special
import utility_mvd
import two_level_decoupling_special


def plot_err_bw(xs, ys, legend, fname, ylim=None, ylabel=r'$\varepsilon$', legend_title=None, base=2):
    line_styles = ('-', '--', '-.', ':')

    assert len(ys) <= len(line_styles)

    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)
    
    for x, y, ls in zip(xs, ys, line_styles):
        ax.plot(x, y, color='k', linestyle=ls)

    print(ylim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.grid()

    #ax.set_xscale('log')
    ax.set_yscale('log', base=base)
    ax.legend(legend, title=legend_title, loc='upper right')

    fig.savefig(fname, transparent=True)
    plt.close()


def setup_problem():
    # div (-k * grad u) + r*u = f
    # k * grad u * n + cu = bc
    #
    # div q + ru = f
    # -q*n + cu = bc
    x, y, nx, ny, t = sympy.symbols('x y nx ny t')
    n = sympy.Matrix([nx, ny])

    k = np.array((
        (1, 3),
        (3, 10)
    ))

    u = sympy.exp(-(1 + 10*t*x)*y**2)
    
    r = 1
    c = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    #f_f = sympy.lambdify([x, y, t], sympy.exp(-(1 + 10*x)*y**2), "numpy")
    #f_f = sympy.lambdify([x, y, t], (1 + 10*x*t)*y**2, "numpy")
    f_f = sympy.lambdify([x, y, t], 50*x*y**2/(1 + t), "numpy")

    return k, r, f_f, c


def split_quadrangles(nodes, quadrangles, inner, boundary):
    """Вырожденные 4-хугольники корректно превращаются в 1 треугольник"""
    diagonal_D_lenghts = np.linalg.norm(nodes[quadrangles[inner, 2]] - nodes[quadrangles[inner, 0]], axis=1)
    diagonal_V_lenghts = np.linalg.norm(nodes[quadrangles[inner, 3]] - nodes[quadrangles[inner, 1]], axis=1)

    triangles_split_D = np.zeros((np.count_nonzero(inner), 2, 3), dtype=quadrangles.dtype)

    # 0, 1, 2
    triangles_split_D[:, 0] = quadrangles[inner, :3]

    # 0, 2, 3
    triangles_split_D[:, 1, 0] = quadrangles[inner, 0]
    triangles_split_D[:, 1, 1:] = quadrangles[inner, 2:]

    triangles_split_V = np.zeros((np.count_nonzero(inner), 2, 3), dtype=quadrangles.dtype)

    # 0, 1, 3
    triangles_split_V[:, 0, :2] = quadrangles[inner, :2]
    triangles_split_V[:, 0, 2] = quadrangles[inner, 3]

    # 1, 2, 3
    triangles_split_V[:, 1] = quadrangles[inner, 1:]

    triangles_inner = np.where((diagonal_D_lenghts < diagonal_V_lenghts)[:, np.newaxis, np.newaxis], triangles_split_D, triangles_split_V)
    triangles_inner = triangles_inner.reshape(-1, 3)

    triangles_boundary = quadrangles[boundary, :3]

    triangles = np.concatenate((triangles_inner, triangles_boundary))
    return triangles


def plot_convergence_v0():
    # время в 4 раза улучшаем, сетку в 2 -> ошибка падает в 4 раза
    #t_degrees = [4 + 2*i for i in range(4)] # 4, 6, 8, 10
    t_degrees = [10 + 2*i for i in range(4)]
    meshes = np.arange(4, 8)

    assert len(t_degrees) == len(meshes)

    #time_meshes = [np.linspace(0, 1, 2**t_degrees[-1] + 1) for t_degree in t_degrees]

    best_mesh_fname = f'meshes/ellipse/quadrangle_{meshes[-1]}.msh'
    path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'article_unsteady' / f'symmetric_m{meshes[-1]}_t{t_degrees[-1]}.npz'
    if not path.is_file():
        time_mesh = np.linspace(0, 1, 2**t_degrees[-2] + 1) # -1 должно быть, но оперативки не хватило
        ys = two_level_special.solve(*setup_problem(), time_mesh, best_mesh_fname, sigma=0.5)
        np.savez_compressed(path, ys=ys)
    
    best_ys = np.load(path)['ys']
    
    nodes, quadrangles = utility_mvd.load_msh(best_mesh_fname)
    groups_best, cells = utility_mvd.load_npz(pathlib.Path(best_mesh_fname).with_suffix('.npz'))
    cell_areas = utility_mvd.compute_cell_areas(cells, groups_best, nodes)

    all_nodes = nodes[:groups_best[3]]

    legend = [rf'${mesh}, 4^{{{-t_degree//2}}}$' for mesh, t_degree in enumerate(t_degrees, 1)]

    ylims = ((4**-7, None), (None, None))

    folder = 'images/unsteady_anisotropic_diffusion_reaction/errors_bw2'
    fnames = (f'{folder}/f-4.pdf', f'{folder}/f-5.pdf')

    for solve, prefix, fname, ylim in zip((two_level_special.solve, two_level_decoupling_special.solve), ('implicit', 'decoupling'), fnames, ylims):
        time_meshes = []
        errs = []
        for mesh, t_degree in zip(meshes[:-1], t_degrees):
            mesh_fname = f'meshes/ellipse/quadrangle_{mesh}.msh'
            path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'article_unsteady' / f'{prefix}_m{mesh}_t{t_degree}.npz'
            time_mesh = np.linspace(0, 1, 2**t_degree + 1)

            if not path.is_file():
                ys = solve(*setup_problem(), time_mesh, mesh_fname)
                np.savez_compressed(path, ys=ys)
            
            ys = np.load(path)['ys']

            # Интерполировать до самой подробной сетки
            # четырехугольники превратить в треугольники и интерполяцию использовать.
            nodes, quadrangles = utility_mvd.load_msh(mesh_fname)
            groups, cells = utility_mvd.load_npz(pathlib.Path(mesh_fname).with_suffix('.npz'))

            boundary = (quadrangles[:, 0] >= groups[2]) & (quadrangles[:, 2] >= groups[2])
            inner = ~boundary

            utility_mvd.redirect_eV_on_boundary(quadrangles, nodes, boundary)

            triangles = split_quadrangles(nodes, quadrangles, inner, boundary)

            triangulation = tri.Triangulation(*nodes[:groups[3]].T, triangles)

            step = 2 ** (t_degrees[-2] - t_degree)
            err = []
            for y, best_y in zip(ys, best_ys[::step]):
                interpolator = tri.LinearTriInterpolator(triangulation, y)
                y = interpolator(*all_nodes.T)

                L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = utility_mvd.compute_errors(y, best_y, cell_areas, groups_best)
                err.append(L2)
            
            time_meshes.append(time_mesh)
            errs.append(err)
        
        plot_err_bw(time_meshes, errs, legend, fname, legend_title=rf'$G, \tau$', base=4, ylim=ylim)


def plot_convergence():
    # время в 4 раза улучшаем, сетку в 2 -> ошибка падает в 4 раза
    #t_degrees = [4 + 2*i for i in range(4)] # 4, 6, 8, 10
    t_degrees = [10 + 2*i for i in range(4)]
    meshes = np.arange(4, 8)

    assert len(t_degrees) == len(meshes)

    #time_meshes = [np.linspace(0, 1, 2**t_degrees[-1] + 1) for t_degree in t_degrees]

    best_mesh_fname = f'meshes/ellipse/quadrangle_{meshes[-1]}.msh'
    path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'article_unsteady' / f'symmetric_m{meshes[-1]}_t{t_degrees[-1]}.npz'
    if not path.is_file():
        time_mesh = np.linspace(0, 1, 2**t_degrees[-2] + 1) # -1 должно быть, но оперативки не хватило
        ys = two_level_special.solve(*setup_problem(), time_mesh, best_mesh_fname, sigma=0.5)
        np.savez_compressed(path, ys=ys)
    
    best_ys = np.load(path)['ys']

    # Интерполировать до самой подробной сетки
    # четырехугольники превратить в треугольники и интерполяцию использовать.
    nodes, quadrangles = utility_mvd.load_msh(best_mesh_fname)
    groups, cells = utility_mvd.load_npz(pathlib.Path(best_mesh_fname).with_suffix('.npz'))

    boundary = (quadrangles[:, 0] >= groups[2]) & (quadrangles[:, 2] >= groups[2])
    inner = ~boundary

    utility_mvd.redirect_eV_on_boundary(quadrangles, nodes, boundary)

    triangles = split_quadrangles(nodes, quadrangles, inner, boundary)

    triangulation = tri.Triangulation(*nodes[:groups[3]].T, triangles)


    legend = [rf'${mesh}, 4^{{{-t_degree//2}}}$' for mesh, t_degree in enumerate(t_degrees, 1)]

    ylims = ((4 ** -7, None), (4 ** -7, 4 ** -3))

    folder = 'images/unsteady_anisotropic_diffusion_reaction/errors_bw2'
    fnames = (f'{folder}/f-4.pdf', f'{folder}/f-5.pdf')

    for solve, prefix, fname, ylim in zip((two_level_special.solve, two_level_decoupling_special.solve), ('implicit', 'decoupling'), fnames, ylims):
        time_meshes = []
        errs = []
        for mesh, t_degree in zip(meshes[:-1], t_degrees):
            mesh_fname = f'meshes/ellipse/quadrangle_{mesh}.msh'
            path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'article_unsteady' / f'{prefix}_m{mesh}_t{t_degree}.npz'
            time_mesh = np.linspace(0, 1, 2**t_degree + 1)

            if not path.is_file():
                ys = solve(*setup_problem(), time_mesh, mesh_fname)
                np.savez_compressed(path, ys=ys)
            
            ys = np.load(path)['ys']

            nodes, quadrangles = utility_mvd.load_msh(mesh_fname)
            groups, cells = utility_mvd.load_npz(pathlib.Path(mesh_fname).with_suffix('.npz'))
            cell_areas = utility_mvd.compute_cell_areas(cells, groups, nodes)

            all_nodes = nodes[:groups[3]]


            step = 2 ** (t_degrees[-2] - t_degree)
            err = []
            for y, best_y in zip(ys, best_ys[::step]):
                interpolator = tri.LinearTriInterpolator(triangulation, best_y)
                u = interpolator(*all_nodes.T)

                L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = utility_mvd.compute_errors(y, u, cell_areas, groups)
                err.append(L2)
            
            time_meshes.append(time_mesh)
            errs.append(err)
        
        plot_err_bw(time_meshes, errs, legend, fname, legend_title=rf'$G, \tau$', base=4, ylim=ylim)


def plot_solutions(prefix='symmetric', mesh=5, t_degree=7, n=12):
    mesh_fname = f'meshes/ellipse/quadrangle_{mesh}.msh'
    path = pathlib.Path('data') / 'anisotropic_diffusion_reaction' / 'article_unsteady' / f'{prefix}_m{mesh}_t{t_degree}.npz'

    # if not path.is_file():
    #     time_mesh = np.linspace(0, 1, 2**t_degree + 1)
    #     ys = two_level_special.solve(*setup_problem(), time_mesh, mesh_fname, sigma=0.5)
    #     np.savez_compressed(path, ys=ys)
    
    time_mesh = np.linspace(0, 1, 2**t_degree + 1)
    ys = two_level_special.solve(*setup_problem(), time_mesh, mesh_fname, sigma=0.5)
    np.savez_compressed(path, ys=ys)
    
    ys = np.load(path)['ys']
    
    indexes = np.linspace(0, ys.shape[0]-1, n).astype(int)
    vmin = ys[indexes].min()
    vmax = ys[indexes].max()

    vmin = ys[indexes[-4:]].min()
    vmax = ys[indexes[-4:]].max()

    vmin = vmax = None

    # четырехугольники превратить в треугольники и интерполяцию использовать.
    nodes, quadrangles = utility_mvd.load_msh(mesh_fname)
    groups, cells = utility_mvd.load_npz(pathlib.Path(mesh_fname).with_suffix('.npz'))

    boundary = (quadrangles[:, 0] >= groups[2]) & (quadrangles[:, 2] >= groups[2])
    inner = ~boundary

    utility_mvd.redirect_eV_on_boundary(quadrangles, nodes, boundary)

    triangles = split_quadrangles(nodes, quadrangles, inner, boundary)

    triangulation = tri.Triangulation(*nodes[:groups[3]].T, triangles)

    nrows = np.floor(10 ** 0.5).astype(int)
    ncols = np.ceil(n / nrows).astype(int)
    fig, axes = plt.subplots(nrows, ncols)

    for i, ax in zip(indexes, axes.flat):
        plot = ax.tripcolor(triangulation, ys[i], vmin=vmin, vmax=vmax)
        ax.set_title(rf'$t = {i / (ys.shape[0] - 1):.3f}$')
        ax.axis('scaled')
        fig.colorbar(plot, ax=ax)

    plt.show()


if __name__ == '__main__':
    folder = 'images/mvd/unsteady'
    plot_convergence()
    #plot_solutions()

    # еще хотим картинки нескольких моментов времени
    # еще там вроде вектор справа на половине слоя)))
