import matplotlib.pyplot as plt
import numpy as np
import utility
import sys
sys.path.append('computations')
import elliptic_mvd
import elliptic_fem


def plot_errornorm(x, y, min_y, max_y, legend, fname, ylabel=r'$\varepsilon$'):
    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns())

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
    
    data_mvd = []
    data_fem = []
    for k in ks:
        data_mvd_i = []
        data_fem_i = []
        for i in range(16):
            data_mvd_i.append(elliptic_mvd.calculate(f'meshes/rectangle/rectangle_{i}_quadrangle', k))
            data_fem_i.append(elliptic_fem.solve(f'meshes/rectangle/rectangle_{i}_triangle.msh', k))
        data_mvd.append(data_mvd_i)
        data_fem.append(data_fem_i)

    data_mvd = np.array(data_mvd)
    data_fem = np.array(data_fem)
    
    L2_min = 1e-5 #min(data_mvd[:, :, 4].min(), data_mvd[:, :, 5].min())
    L2_max = 2e-2#max(data_mvd[:, :, 4].max(), data_mvd[:, :, 5].max())
    Lmax_min = 1e-4 #min(data_mvd[:, :, 1].min(), data_mvd[:, :, 2].min())
    Lmax_max = 4e-2 # min(data_mvd[:, :, 1].max(), data_mvd[:, :, 2].max())

    for j, (data_mvd_i, data_fem_i) in enumerate(zip(data_mvd, data_fem), 1):
        plot_errornorm(data_mvd_i[:, 0], data_mvd_i[:, 4:6], L2_min, L2_max,
                       (r'$D$-grid', r'$V$-grid'), f'images/mvd/k{j}-1.pdf', ylabel=r'$\varepsilon_2$')
        
        plot_errornorm(data_mvd_i[:, 0], data_mvd_i[:, 1:3], Lmax_min, Lmax_max,
                       (r'$D$-grid', r'$V$-grid'), f'images/mvd/k{j}-2.pdf', ylabel=r'$\varepsilon_\infty$')
     
    # L2_min = 1e-5
    # L2_max = 1e-2
    # Lmax_min = 1e-4
    # Lmax_max = 1e-2

    # dofs_num, node_num, cell_num, error_L2, error_max, E_H10
    # element_tags, L_max_D, L_max_V, L_max, L_D, L_V, L, vector_errornorm
       


    