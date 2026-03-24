import matplotlib.pyplot as plt
import numpy as np
import utility
import os
import sympy
import sys
sys.path.append('computations')
import elliptic_mvd_sparse_Dirichlet
import elliptic_mvd_sparse_Robin


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


def mvd_Dirichlet(k, prefix):
    data = []
    for i in range(1, 9):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data.append(elliptic_mvd_sparse_Dirichlet.calculate(mesh, k))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    
    data = np.array(data)
    x = data[:, 0]

    plot_err(x, data[:, 1:4], ('L2_D', 'L2_V', 'L2'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err(x, data[:, 4:6], ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err(x, data[:, 7], ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def mvd_Neumann0(k, prefix):
    data = []
    for i in range(1, 9):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data.append(elliptic_mvd_sparse_Neumann.calculate(mesh, k))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    
    data = np.array(data)
    x = data[:, 0]

    plot_err(x, data[:, 1:4], ('L2_D', 'L2_V', 'L2'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err(x, data[:, 4:6], ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err(x, data[:, 7], ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def mvd_Neumann(k, prefix):
    data = []
    for i in range(1, 9):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 0))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    
    data = np.array(data)
    x = data[:, 0]

    plot_err(x, data[:, 1:4], ('L2_D', 'L2_V', 'L2'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err(x, data[:, 4:6], ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err(x, data[:, 7], ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def mvd_Robin(k, prefix, c=1):
    data = []
    for i in range(1, 9):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, c))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    
    data = np.array(data)
    x = data[:, 0]

    plot_err(x, data[:, 1:4], ('L2_D', 'L2_V', 'L2'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err(x, data[:, 4:6], ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err(x, data[:, 7], ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def Neumann_compare(k, prefix):
    data1 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data1.append(elliptic_mvd_sparse_Neumann.calculate(mesh, k))
    data1 = np.array(data1)

    data2 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data2.append(elliptic_mvd_sparse_Robin_v0.calculate(mesh, k, 0))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data2 = np.array(data2)

    data3 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data3.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 0))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data3 = np.array(data3)
    x = data3[:, 0]

    plot_err_compare(x, (data1[:, 1:3], data2[:, 1:3], data3[:, 1:3]), ('L2_D', 'L2_V'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err_compare(x, (data1[:, 4:6], data2[:, 4:6], data3[:, 4:6]), ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err_compare(x, (data1[:, 7].reshape(-1, 1), data2[:, 7][:, np.newaxis], data3[:, 7][:, np.newaxis]), ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def Robin_compare(k, prefix):
    data1 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data1.append(elliptic_mvd_sparse_Robin_v0.calculate(mesh, k, 1))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data1 = np.array(data1)

    data2 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data2.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 1))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data2 = np.array(data2)
    x = data2[:, 0]

    plot_err_compare(x, (data1[:, 1:3], data2[:, 1:3]), ('L2_D', 'L2_V'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err_compare(x, (data1[:, 4:6], data2[:, 4:6]), ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err_compare(x, (data1[:, 7].reshape(-1, 1), data2[:, 7][:, np.newaxis]), ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def Robin_and_Dirichlet_compare(k, prefix):
    data1 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data1.append(elliptic_mvd_sparse_Dirichlet.calculate(mesh, k))
    data1 = np.array(data1)

    data2 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data2.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 1))    
    data2 = np.array(data2)

    data3 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data3.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 100))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data3 = np.array(data3)
    x = data3[:, 0]

    data4 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data4.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 10000))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data4 = np.array(data4)
    x = data4[:, 0]

    plot_err_compare(x, (data1[:, 1:3], data2[:, 1:3], data3[:, 1:3], data4[:, 1:3]), ('L2_D', 'L2_V'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err_compare(x, (data1[:, 4:6], data2[:, 4:6], data3[:, 4:6], data4[:, 4:6]), ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err_compare(x, (data1[:, 7].reshape(-1, 1), data2[:, 7].reshape(-1, 1), data3[:, 7].reshape(-1, 1), data4[:, 7].reshape(-1, 1)), ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def all_bc_compare(k, prefix, c=1):
    data1 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data1.append(elliptic_mvd_sparse_Dirichlet.calculate(mesh, k))
    data1 = np.array(data1)

    data2 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data2.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, 0))    
    data2 = np.array(data2)

    data3 = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle'
        data3.append(elliptic_mvd_sparse_Robin.calculate(mesh, k, c))
        # coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    data3 = np.array(data3)
    x = data3[:, 0]

    plot_err_compare(x, (data1[:, 1:3], data2[:, 1:3], data3[:, 1:3]), ('L2_D', 'L2_V'), f'{prefix}_L2.pdf', ylabel=r'$\varepsilon_2$')
    plot_err_compare(x, (data1[:, 4:6], data2[:, 4:6], data3[:, 4:6]), ('Lmax_D', 'Lmax_V'), f'{prefix}_Lmax.pdf', ylabel=r'$\varepsilon_\infty$')
    plot_err_compare(x, (data1[:, 7].reshape(-1, 1), data2[:, 7].reshape(-1, 1), data3[:, 7].reshape(-1, 1)), ('L2_q',), f'{prefix}_L2_q.pdf', ylabel=r'$\varepsilon$')


def setup_problem_Neumann_scalar():
    x, y, nx, ny = sympy.symbols('x y nx ny')
    n = sympy.Matrix([nx, ny])

    u = sympy.exp(x*y)
    k = 1
    r = 1
    c = 0

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) - c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q


def setup_problem_Robin_scalar():
    x, y, nx, ny = sympy.symbols('x y nx ny')
    n = sympy.Matrix([nx, ny])

    u = sympy.exp(x*y)
    k = 1
    r = 1
    c = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) - c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q


def setup_problem_Dirichlet():
    x, y = sympy.symbols('x y')

    u = sympy.exp(x*y)
    k = sympy.Matrix([
        [1, 3],
        [3, 10]
    ])
    r = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y], u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, bc, u, q

def setup_problem_Robin(c=1):
    x, y, nx, ny = sympy.symbols('x y nx ny')
    n = sympy.Matrix([nx, ny])

    u = sympy.exp(x*y)
    k = sympy.Matrix([
        [1, 3],
        [3, 10]
    ])
    r = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) - c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q

def setup_problem_Neumann():
    return setup_problem_Robin(c=0)


def investigate_problem(solve, setup, fname):
    data = []
    for i in range(1, 11):
        mesh = f'meshes/rectangle/rectangle_{i}_quadrangle.msh'
        data.append(solve(*setup(), mesh))
    data = np.array(data)
    # nodes.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    x = data[:, 0]
    plot_err(x, (data[:, np.r_[1:3, 4:6, 7]]), ('$L_2^D$', '$L_2^V$', '$L_{max}^D$', '$L_{max}^V$', '$L_2^q$'), fname, ylabel=r'$\varepsilon$')


def setup_problem():
    x, y = sympy.symbols('x y')
    u = sympy.exp(x*y)
    k = sympy.Matrix([
        [1, 3],
        [3, 10]
    ])
    r = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)
    f = div + r*u

    return x, y, k, r, f, u, q


def setup_problem_Dirichlet():
    x, y, k, r, f, u, q = setup_problem()

    f = sympy.lambdify([x, y], f, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, u, q


def setup_problem_Robin(c=1):
    x, y, k, r, f, u, q = setup_problem()
    nx, ny = sympy.symbols('nx ny')
    n = sympy.Matrix([nx, ny])

    f = sympy.lambdify([x, y], f, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) - c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q


def setup_problem_Robin_as_fake_Dirichlet(c=1):
    x, y, k, r, f, u, q = setup_problem()
    nx, ny = sympy.symbols('nx ny')
    n = sympy.Matrix([nx, ny])

    f = sympy.lambdify([x, y], f, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], -c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q


def investigate_fake_Dirichlet(folder):
    n = 9

    data_D = []
    for i in range(1, n):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_quadrangle.msh'
        data_D.append(elliptic_mvd_sparse_Dirichlet.solve_problem(*setup_problem_Dirichlet(), mesh_fname))
    data_D = np.array(data_D)
    
    cs_fake = [2e7 / 5 ** i for i in range(5)]
    cs_fake.reverse()
    data_fake_R = []
    for i in range(1, n):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_quadrangle.msh'
        d = []
        for c in cs_fake:
            d.append(elliptic_mvd_sparse_Robin.solve_problem(*setup_problem_Robin_as_fake_Dirichlet(c), mesh_fname))
        data_fake_R.append(d)
    data_fake_R = np.transpose(np.array(data_fake_R), axes=(1, 0, 2))

    cs = [2000 / 5 ** i for i in range(5)]
    cs.reverse()
    data_R = []
    for i in range(1, n):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_quadrangle.msh'
        d = []
        for c in cs:
            d.append(elliptic_mvd_sparse_Robin.solve_problem(*setup_problem_Robin(c), mesh_fname))
        data_R.append(d)
    data_R = np.transpose(np.array(data_R), axes=(1, 0, 2))
        
    # nodes.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q
    x = data_D[:, 0]

    legend_cs_fake = (*(f'fake Robin $c={c:.2e}$' for c in cs_fake), 'Dirichlet')
    plot_err(x, np.column_stack((*data_fake_R[:, :, 3], data_D[:, 3])), legend_cs_fake, os.path.join(folder, 'fake_Robin_L2.pdf'))
    plot_err(x, np.column_stack((*data_fake_R[:, :, 6], data_D[:, 6])), legend_cs_fake, os.path.join(folder, 'fake_Robin_Lmax.pdf'))
    plot_err(x, np.column_stack((*data_fake_R[:, :, 7], data_D[:, 7])), legend_cs_fake, os.path.join(folder, 'fake_Robin_L2_q.pdf'))

    legend_cs = (*(f'Robin $c={c}$' for c in cs), 'Dirichlet')
    plot_err(x, np.column_stack((*data_R[:, :, 3], data_D[:, 3])), legend_cs, os.path.join(folder, 'Robin_L2.pdf'))
    plot_err(x, np.column_stack((*data_R[:, :, 6], data_D[:, 6])), legend_cs, os.path.join(folder, 'Robin_Lmax.pdf'))
    plot_err(x, np.column_stack((*data_R[:, :, 7], data_D[:, 7])), legend_cs, os.path.join(folder, 'Robin_L2_q.pdf'))


if __name__ == '__main__':
    k = np.array(
        ((1, 9),
         (9, 100))
    )

    folder = 'images/mvd/boundary_conditions'

    investigate_fake_Dirichlet('images/mvd/boundary_conditions/fake_Dirichlet')

    

    # investigate_scalar_problem(elliptic_mvd_sparse_Robin_new_scalar.solve_problem, setup_problem_Neumann_scalar, f'{folder}/scalar_Neumann.pdf')
    # investigate_scalar_problem(elliptic_mvd_sparse_Robin_new_scalar.solve_problem, setup_problem_Robin_scalar, f'{folder}/scalar_Robin.pdf')

    # investigate_problem(elliptic_mvd_sparse_Robin_new_matrix.solve_problem, setup_problem_Neumann, f'{folder}/matrix_Neumann.pdf')
    # investigate_problem(elliptic_mvd_sparse_Robin_new_matrix.solve_problem, setup_problem_Robin, f'{folder}/matrix_Robin.pdf')


    # mvd_Dirichlet(k, f'{folder}/Dirichlet')
    # mvd_Neumann0(k, f'{folder}/Neumann0')
    # mvd_Neumann(k, f'{folder}/Neumann')
    # mvd_Robin(k, f'{folder}/Robin')

    #Neumann_compare(k, f'{folder}/compare_Neumann')

    #Robin_compare(k, f'{folder}/compare_Robin')

    #all_bc_compare(k, f'{folder}/full_compare')

    #Robin_and_Dirichlet_compare(k, f'{folder}/compare_Robin_and_Dirichlet')
    