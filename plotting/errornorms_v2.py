import matplotlib.pyplot as plt
import numpy as np
import utility
import sys
sys.path.append('computations')
import elliptic_mvd
import elliptic_mvd_sparse
import elliptic_fem


def plot_errornorm(x, y, min_y, max_y, legend, fname, ylabel=r'$\varepsilon$'):
    #fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns())
    fig, ax = plt.subplots(figsize=utility.get_default_figsize())

    ax.plot(x, y, 'o-')

    ax.set_ylim(min_y, max_y)
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(legend)

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)

    plt.close()


if __name__ == '__main__':
    ks = np.array((
        ((1, 0),
         (0, 1)),
         ((1, 0),
         (0, 100)),
         ((1, 9),
         (9, 100)),
         ((1, -9),
         (-9, 100)),
    ))
    
    # data_mvd = []
    # #data_fem = []
    # for k in ks:
    #     data_mvd_i = []
    #     data_fem_i = []
    #     for i in range(22):
    #         data_mvd_i.append(elliptic_mvd_sparse.calculate(f'meshes/rectangle/rectangle_{i}_quadrangle', k, info=True))
    #         #data_fem_i.append(elliptic_fem.solve(f'meshes/rectangle/rectangle_{i}_triangle.msh', k))
    #     data_mvd.append(data_mvd_i)
    #     #data_fem.append(data_fem_i)

    # data_mvd = np.array(data_mvd)
    # np.savez_compressed('data.npz', data_mvd=data_mvd)
    # exit()

    data_mvd = np.load('data.npz')['data_mvd']
    #data_fem = np.array(data_fem)
    
    # L2_min = 1e-5
    # L2_max = 2e-2
    # Lmax_min = 1e-4
    # Lmax_max = 4e-2

    L2_min = min(data_mvd[:, :, 4].min(), data_mvd[:, :, 5].min())
    L2_max = max(data_mvd[:, :, 4].max(), data_mvd[:, :, 5].max())

    Lmax_min = min(data_mvd[:, :, 1].min(), data_mvd[:, :, 2].min())
    Lmax_max = max(data_mvd[:, :, 1].max(), data_mvd[:, :, 2].max())

    Lmvd_min = data_mvd[:, :, 7].min()
    Lmvd_max = data_mvd[:, :, 7].max()

    L2_min = 10 ** np.floor(np.log10(L2_min))
    L2_max = 10 ** np.ceil(np.log10(L2_max))

    Lmax_min = 10 ** np.floor(np.log10(Lmax_min))
    Lmax_max = 10 ** np.ceil(np.log10(Lmax_max))

    Lmvd_min = 10 ** np.floor(np.log10(Lmvd_min))
    Lmvd_max = 10 ** np.ceil(np.log10(Lmvd_max))

    #L2_min, L2_max = 1e-5, 1#7e-5, 3e-1
    #Lmax_min, Lmax_max = 1e-4, 1
    #Lmvd_min, Lmvd_max = 1e-4, 10

    # node_coords.shape[0], L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2, vector_errornorm

    legend = (r'$K_1$', r'$K_2$', r'$K_3$', r'$K_4$')

    plot_errornorm(data_mvd[0, :, 0], data_mvd[:, :, 4].T, L2_min, L2_max,
                       legend, f'images/mvd/err_2_D.pdf', ylabel=r'$\varepsilon_2^D$')
    
    plot_errornorm(data_mvd[0, :, 0], data_mvd[:, :, 5].T, L2_min, L2_max,
                       legend, f'images/mvd/err_2_V.pdf', ylabel=r'$\varepsilon_2^V$')
    
    plot_errornorm(data_mvd[0, :, 0], data_mvd[:, :, 1].T, Lmax_min, Lmax_max,
                       legend, f'images/mvd/err_max_D.pdf', ylabel=r'$\varepsilon_\infty^D$')
    
    plot_errornorm(data_mvd[0, :, 0], data_mvd[:, :, 2].T, Lmax_min, Lmax_max,
                       legend, f'images/mvd/err_max_V.pdf', ylabel=r'$\varepsilon_\infty^V$')
    
    plot_errornorm(data_mvd[0, :, 0], data_mvd[:, :, 7].T, Lmvd_min, Lmvd_max,
                       legend, f'images/mvd/err_2_MVD.pdf', ylabel=r'$\varepsilon$')
    
    