import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh
import sys
sys.path.append('computations/test_fe')
import poisson_fvm_Robin
import poisson_fem_Robin


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.size'] = 12


def read_msh(mesh_fname):
    gmsh.initialize()
    gmsh.open(mesh_fname)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)
    width = xmax - xmin
    height = ymax - ymin

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    triangle_nodes = triangle_nodes.reshape(-1, 3) - 1

    cell_edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type)
    gmsh.model.mesh.create_edges()
    cell_edges, edge_orientations = gmsh.model.mesh.get_edges(cell_edge_nodes)
    edges, edge_nodes = gmsh.model.mesh.getAllEdges()
    edge_nodes = edge_nodes.reshape(-1, 2) - 1
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    return width, height, node_coords, triangle_nodes, edge_nodes


def finite_volume(mesh_fname, fname):
    width, height, node_coords, triangle_nodes, edge_nodes = read_msh(mesh_fname)

    pc_cells = PolyCollection(node_coords[triangle_nodes], closed=True, facecolors='none', edgecolors='b')

    triangle_centers = node_coords[triangle_nodes].sum(axis=1) / 3
    inner_volume = np.stack((node_coords[10], triangle_centers[12], node_coords[13], triangle_centers[16]))
    boundary_volume = np.stack((node_coords[9], triangle_centers[9], node_coords[8]))
    pc_volumes = PolyCollection((inner_volume, boundary_volume), closed=True, facecolors=(0, 1, 0, 0.3), edgecolors='g')

    # finite_volumes
    fig, ax = plt.subplots(figsize=utility.get_figsize(width, height), constrained_layout=True)

    ax.add_collection(pc_cells)
    ax.add_collection(pc_volumes)

    edge_centers = node_coords[edge_nodes].sum(axis=1) / 2
    ax.plot(*edge_centers.T, 'ob')

    # for node_tag, node_coord in zip(node_tags - 1, node_coords):
    #     plt.text(*node_coord, node_tag, size=14)
    
    # triangle_centers = node_coords[triangle_nodes].sum(axis=1) / 3
    # for triangle_tag, center in zip(range(triangle_centers.shape[0]), triangle_centers):
    #     plt.text(*center, triangle_tag, size=14)
    
    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()    


def grad_approximation(mesh_fname, fname):
    width, height, node_coords, triangle_nodes, edge_nodes = read_msh(mesh_fname)

    edge_centers = (node_coords[triangle_nodes[12]] + node_coords[np.roll(triangle_nodes[12], -1)]) / 2

    pc_triangle = PolyCollection([node_coords[triangle_nodes[12]]], closed=True, facecolors='none', edgecolors='b')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    ax.add_collection(pc_triangle)
    ax.plot(*edge_centers.T, 'ob')

    step = 0.005
    plt.text(edge_centers[0, 0] + step, edge_centers[0, 1] - 1 * step, fr'$\bm x_{1}$')
    plt.text(edge_centers[1, 0] + step, edge_centers[1, 1] + 1 * step, fr'$\bm x_{2}$')
    plt.text(edge_centers[2, 0] + step, edge_centers[2, 1] + 0 * step, fr'$\bm x_{3}$')
    
    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def div_approximation(mesh_fname, fname):
    width, height, node_coords, triangle_nodes, edge_nodes = read_msh(mesh_fname)

    edge_centers_1 = (node_coords[triangle_nodes[12]] + node_coords[np.roll(triangle_nodes[12], -1)]) / 2
    edge_centers_2 = (node_coords[triangle_nodes[16]] + node_coords[np.roll(triangle_nodes[16], -1)]) / 2
    edge_centers = np.concatenate((edge_centers_1, edge_centers_2[::2]))

    triangle_centers = node_coords[triangle_nodes].sum(axis=1) / 3
    inner_volume = np.stack((node_coords[10], triangle_centers[12], node_coords[13], triangle_centers[16]))
    pc_volumes = PolyCollection((inner_volume,), closed=True, facecolors=(0, 1, 0, 0.3), edgecolors='g')

    pc_triangles = PolyCollection(np.stack((node_coords[triangle_nodes[12]], node_coords[triangle_nodes[16]])), closed=True, facecolors='none', edgecolors='b')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    ax.add_collection(pc_triangles)
    ax.add_collection(pc_volumes)
    ax.plot(*edge_centers.T, 'ob')

    step = 0.005
    plt.text(edge_centers[0, 0] + step, edge_centers[0, 1] - 1 * step, fr'$\bm x_{1}$')
    plt.text(edge_centers[1, 0] + step, edge_centers[1, 1] + 1 * step, fr'$\bm x_{2}$')
    plt.text(edge_centers[2, 0] + step, edge_centers[2, 1] + 0 * step, fr'$\bm x_{3}$')
    plt.text(edge_centers[3, 0] + step, edge_centers[3, 1] + 0 * step, fr'$\bm x_{5}$')
    plt.text(edge_centers[4, 0] + step, edge_centers[4, 1] + 1 * step, fr'$\bm x_{4}$')
    
    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def boundary_conditions(mesh_fname, fname):
    width, height, node_coords, triangle_nodes, edge_nodes = read_msh(mesh_fname)

    edge_centers = (node_coords[triangle_nodes[9]] + node_coords[np.roll(triangle_nodes[9], -1)]) / 2

    triangle_centers = node_coords[triangle_nodes].sum(axis=1) / 3
    boundary_volume = np.stack((node_coords[9], triangle_centers[9], node_coords[8]))
    pc_volumes = PolyCollection((boundary_volume,), closed=True, facecolors=(0, 1, 0, 0.3), edgecolors='g')

    pc_triangles = PolyCollection((node_coords[triangle_nodes[9]],), closed=True, facecolors='none', edgecolors='b')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    ax.add_collection(pc_triangles)
    ax.add_collection(pc_volumes)
    ax.plot(*edge_centers.T, 'ob')

    step = 0.005
    plt.text(edge_centers[0, 0] + 1 * step, edge_centers[0, 1] - 3 * step, fr'$\bm x_{1}$')
    plt.text(edge_centers[1, 0] + step, edge_centers[1, 1] + 1 * step, fr'$\bm x_{2}$')
    plt.text(edge_centers[2, 0] - 5 * step, edge_centers[2, 1] + 1 * step, fr'$\bm x_{3}$')
    
    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def different_centers_compare(K1, K2, fnames, n=10):
    data = []
    for i in range(1, n+1):
        data.append([])
        for K in K1, K2:
            data[-1].append([])
            for center in (poisson_fvm_Robin.centroid, poisson_fvm_Robin.circumcenter, poisson_fvm_Robin.incenter, poisson_fvm_Robin.orthocenter):
                mesh_fname = f'meshes/rectangle/rectangle_{i}_triangle.msh'
                results = poisson_fvm_Robin.solve(*poisson_fvm_Robin.setup_problem(K, 1, 1), center, mesh_fname)
                data[-1][-1].append(results)
    data = np.array(data)

    for index, (ylabel, fname) in enumerate(zip((r'$\varepsilon_2$', r'$\varepsilon_\infty$'), fnames), 1):

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=utility.get_default_figsize(), constrained_layout=True)

        for k, ax in enumerate(axs):
            ms = 12
            for i in range(4):
                ax.plot(data[:, k, i, 0], data[:, k, i, index], 'o-', ms=ms, lw=ms/4)
                ms -= 2

            ax.set_xlabel('$N$')
            ax.grid()
            ax.loglog()
        axs[0].set_ylabel(ylabel)
                    
        fig.legend(('centroid', 'circumcenter', 'incenter', 'orthocenter'))

        fig.savefig(fname, transparent=True)
        plt.close()


def compare_fvm_and_fem(Ks, fnames, n=10):
    data = []
    for i in range(1, n+1):
        data.append([])
        for K in Ks:
            data[-1].append([])
            mesh_fname = f'meshes/rectangle/rectangle_{i}_triangle.msh'
            results_fvm = poisson_fvm_Robin.solve(*poisson_fvm_Robin.setup_problem(K, 1, 1), poisson_fvm_Robin.centroid, mesh_fname)
            results_fem = poisson_fem_Robin.solve(mesh_fname, ('CR', 1), K, 1, 1)
            data[-1][-1].append(results_fvm)
            data[-1][-1].append(results_fem)
    
    data = np.array(data)

    for index, (ylabel, fname) in enumerate(zip((r'$\varepsilon_2$', r'$\varepsilon_\infty$'), fnames), 1):

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=utility.get_default_figsize(), constrained_layout=True)

        for k, ax in enumerate(axs):
            ms = 9
            ax.plot(data[:, k, 0, 0], data[:, k, 0, index], 'o-', ms=ms, lw=ms/4)
            ms -= 3
            ax.plot(data[:, k, 1, 0], data[:, k, 1, index], 'o-', ms=ms, lw=ms/4)

            ax.set_xlabel('$N$')
            ax.grid()
            ax.loglog()
        axs[0].set_ylabel(ylabel)
                    
        fig.legend(('FVM', 'FEM'))

        fig.savefig(fname, transparent=True)
        plt.close()




if __name__ == '__main__':
    # finite_volume(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/finite_volumes.pdf')
    # grad_approximation(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/grad_approximation.pdf')
    # div_approximation(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/div_approximation.pdf')
    # boundary_conditions(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/boundary_conditions.pdf')

    K1 = np.array((
        (1, 0),
        (0, 1),
    ))
    K2 = np.array((
        (1, 3),
        (3, 10),
    ))
    # different_centers_compare(K1, 'images/fvm_edge/different_centers_compare_K11.pdf')
    # different_centers_compare(K2, 'images/fvm_edge/different_centers_compare_K21.pdf')
    fnames = (
        'images/fvm_edge/different_centers_compare_L2.pdf',
        'images/fvm_edge/different_centers_compare_Lmax.pdf'
    )
    #different_centers_compare(K1, K2, fnames)

    fnames = (
        'images/fvm_edge/compare_fvm_and_fem_L2.pdf',
        'images/fvm_edge/compare_fvm_and_fem_Lmax.pdf'
    )
    compare_fvm_and_fem((K1, K2), fnames, 7)
