import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('computations')
import elliptic_mvd
import elliptic_fem


def plot_errornorm(x, y, legend, fname, ylabel=r'$\varepsilon$'):
    fig, ax = plt.subplots()

    ax.plot(x, y, 'o-')
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(legend)

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout(pad=0.2)
    #fig.savefig(fname, transparent=True)
    
    #plt.close()


if __name__ == '__main__':
    k = np.array(
        ((1, -9),
         (-9, 100))
    )

    data = []
    data_fem = []
    for i in range(10):
        data.append(elliptic_mvd.calculate(f'meshes/rectangle/rectangle_{i}_quadrangle', k))
        data_fem.append(elliptic_fem.solve(f'meshes/rectangle/rectangle_{i}_triangle.msh', k))
    data = np.array(data)
    data_fem = np.array(data_fem)
    # dofs_num, node_num, cell_num, error_L2, error_max, E_H10
    # element_tags, L_max_D, L_max_V, L_max, L_D, L_V, L
    plot_errornorm(data[:, 0], np.concatenate((data[:, 1:3], data_fem[:, 4].reshape(-1, 1)), axis=1), (r'$L_\infty^D$', r'$L_\infty^V$', r'$L_\infty^{FEM}$'), 'images/mvd/L_max_K3.pdf')
    plot_errornorm(data[:, 0], np.concatenate((data[:, 4:], data_fem[:, 3].reshape(-1, 1)), axis=1), (r'$L_2^D$', r'$L_2^V$', r'$L_2$', r'$L_2^{FEM}$'), 'images/mvd/L_2_K3.pdf')

    plt.show()

    # print(data[:, 0])
    # print(data_fem[:, 0])
    # print(data_fem[:, 1])

    # print(data[:, 1])
    # print(data[:, 2])
    # print(data[:, 4])
    # print(data[:, 5])

    # сравнить матрица
    # сравнить вектора решений
    