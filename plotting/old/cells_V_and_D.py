import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import utility
import gmsh


def plot_triangle_mesh(quadrangle_mesh, fname):
    figsize, node_coords, cell_nodes, voronoi_cells, voronoi_cells_coords, voronoi_nodes_coords, triangle_cells, triangle_cells_coords, triangle_nodes_coords = utility.read_quad_mesh(quadrangle_mesh)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    plt.rcParams['font.size'] = 12

    #print(voronoi_cells)
    #print(cell_nodes)

    fc = [(1, 0, 0, 0.3) if np.array_equal(cell_nodes[3], current_cell_nodes) or np.array_equal(cell_nodes[27], current_cell_nodes) else 'none' for current_cell_nodes in voronoi_cells]
    #fc = [(1, 0, 0, 0.3) if np.array_equal(cell_nodes[3], current_cell_nodes) else 'none' for current_cell_nodes in voronoi_cells]
    pc_V = matplotlib.collections.PolyCollection(voronoi_cells_coords, facecolors=fc, edgecolors='r')

    fc = [(0, 0, 1, 0.3) if np.array_equal(cell_nodes[17], current_cell_nodes) else 'none' for current_cell_nodes in triangle_cells]
    pc_D = matplotlib.collections.PolyCollection(triangle_cells_coords, facecolors=fc, edgecolors='b')

    fig, ax = plt.subplots(figsize=figsize)

    #for i, (xi,yi) in enumerate(node_coords):
    #    plt.text(xi,yi,i)#, size=8)

    ax.add_collection(pc_V)
    ax.add_collection(pc_D)

    ax.plot(voronoi_nodes_coords[:, 0], voronoi_nodes_coords[:, 1], 'or')
    ax.plot(triangle_nodes_coords[:, 0], triangle_nodes_coords[:, 1], 'ob')

    plt.text(node_coords[3, 0], node_coords[3, 1] + 0.04, r'$\bm x_i^D$')
    plt.text(node_coords[3, 0] - 0.06, node_coords[3, 1] - 0.12, r'$\Omega_i^D$')

    plt.text(node_coords[27, 0] + 0.015, node_coords[27, 1], r'$\bm x_{i+}^D$')
    plt.text(node_coords[27, 0] - 0.06, node_coords[27, 1] - 0.12, r'$\Omega_{i+}^D$')

    plt.text(node_coords[17, 0] + 0.014, node_coords[17, 1] - 0.04, r'$\bm x_j^V$')
    plt.text(node_coords[17, 0] - 0.08, node_coords[17, 1] - 0.08, r'$\Omega_j^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot_triangle_mesh(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/cells_V_and_D.pdf')
