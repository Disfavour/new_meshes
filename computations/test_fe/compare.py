import poisson_fem
import poisson_fvm
import numpy as np
import basix.ufl
import matplotlib.pyplot as plt



def plot_compare_results():
    n_mesh = 11 # 11

    # K = np.array((
    #     (1, 3),
    #     (3, 10)
    # ))

    K = np.array((
        (1, 0),
        (0, 1)
    ))

    data_fem = []
    for i in range(1, n_mesh):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_triangle.msh'
        for finite_element in (('Lagrange', 1), ('CR', 1)):
            res = poisson_fem.solve(mesh_fname, finite_element, K)
            print(res)
            
            data_fem.append(res)
    
    data_fem = np.array(data_fem)
    # return num_dofs_global, num_nodes_global, num_cells_global, error_L2, error_max


    data_fvm = []
    for i in range(1, n_mesh):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_triangle.msh'
        for center in (poisson_fvm.centroid, poisson_fvm.circumcenter):
            res = poisson_fvm.solve(mesh_fname, *poisson_fvm.setup_problem(K), center, K)
            print(res)
            
            data_fvm.append(res)
    
    data_fvm = np.array(data_fvm)
    # return edges.size, triangle_mean_area, areas.mean(), L2, Lmax


    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5), layout='constrained')

    axs[0].plot(data_fem[::2, 0], data_fem[::2, 3], '-o')
    axs[0].plot(data_fem[1::2, 0], data_fem[1::2, 3], '-o')
    axs[0].plot(data_fvm[::2, 0], data_fvm[::2, 3], '-o')
    axs[0].plot(data_fvm[1::2, 0], data_fvm[1::2, 3], '-o')
    axs[0].set_ylabel(r'$L_2$')
    axs[0].set_xlabel('$N$')

    axs[1].plot(data_fem[::2, 0], data_fem[::2, 4], '-o')
    axs[1].plot(data_fem[1::2, 0], data_fem[1::2, 4], '-o', ms=10)
    axs[1].plot(data_fvm[::2, 0], data_fvm[::2, 4], '-o')
    axs[1].plot(data_fvm[1::2, 0], data_fvm[1::2, 4], '-o')
    axs[1].set_ylabel(r'$L_\infty$')
    axs[1].set_xlabel('$N$')

    for ax in axs.flat:
        ax.grid()
        ax.loglog()
        ax.legend(('FEM Lagrage', 'FEM CR', 'FVM centroid', 'FVM circumcenter'))
    
    plt.savefig('compare_fem_fvm_K1_1.pdf', transparent=True)


    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5), layout='constrained')

    axs[0].plot(data_fvm[::2, 1], data_fem[::2, 3], '-o')
    axs[0].plot(data_fvm[1::2, 1], data_fem[1::2, 3], '-o')
    axs[0].plot(data_fvm[::2, 2], data_fvm[::2, 3], '-o')
    axs[0].plot(data_fvm[1::2, 2], data_fvm[1::2, 3], '-o')
    axs[0].set_ylabel(r'$L_2$')
    axs[0].set_xlabel('$h$')

    axs[1].plot(data_fvm[::2, 1], data_fem[::2, 4], '-o')
    axs[1].plot(data_fvm[1::2, 1], data_fem[1::2, 4], '-o')
    axs[1].plot(data_fvm[::2, 2], data_fvm[::2, 4], '-o')
    axs[1].plot(data_fvm[1::2, 2], data_fvm[1::2, 4], '-o')
    axs[1].set_ylabel(r'$L_\infty$')
    axs[1].set_xlabel('$h$')

    for ax in axs.flat:
        ax.grid()
        ax.loglog()
        ax.legend(('FEM Lagrage', 'FEM CR', 'FVM centroid', 'FVM circumcenter'))
        ax.invert_xaxis()
    
    plt.savefig('compare_fem_fvm_K1_2.pdf', transparent=True)

    

if __name__ == '__main__':
    plot_compare_results()
