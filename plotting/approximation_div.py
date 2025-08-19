import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh


def plot(quadrangle_mesh, fname):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    plt.rcParams['font.size'] = 12

    gmsh.initialize()
    gmsh.open(f'{quadrangle_mesh}.msh')
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]

    quadrangle_tags, quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    quadrangle_nodes = quadrangle_nodes.reshape(-1, 4) - 1
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    loaded = np.load(f'{quadrangle_mesh}.npz', allow_pickle=True)
    node_groups = loaded['node_groups'].astype(int)
    cell_nodes = loaded['cells'] - 1

    cells_D = np.concatenate((cell_nodes[:node_groups[0]], cell_nodes[node_groups[1]:node_groups[2]]))
    cells_D_coords = [node_coords[cell] for cell in cells_D]

    nodes_D = np.concatenate((np.arange(node_groups[0]), np.arange(node_groups[1], node_groups[2])))
    nodes_D_coords = node_coords[nodes_D]

    cells_V = np.stack((*cell_nodes[node_groups[0]:node_groups[1]], *cell_nodes[node_groups[2]:node_groups[3]]))
    cells_V_coords = node_coords[cells_V]

    nodes_V = np.concatenate((np.arange(node_groups[0], node_groups[1]), np.arange(node_groups[2], node_groups[3])))
    nodes_V_coords = node_coords[nodes_V]

    # cell_V 17
    triangle = cell_nodes[17]
    triangle_coords = node_coords[triangle]

    quads = [27, 28,29]
    quad_nodes = quadrangle_nodes[quads]
    quad_nodes_coords = node_coords[quad_nodes]

    e1 = quad_nodes_coords[:, 2] - quad_nodes_coords[:, 0]
    e2 = quad_nodes_coords[:, 3] - quad_nodes_coords[:, 1]
    e1 /= np.linalg.norm(e1, axis=1).reshape(3, -1)
    e2 /= np.linalg.norm(e2, axis=1).reshape(3, -1)

    edge_centers = np.stack((triangle_coords[:2].sum(axis=0)/2, triangle_coords[1:].sum(axis=0)/2, triangle_coords[::2].sum(axis=0)/2))

    pc_triangle = PolyCollection([triangle_coords], facecolors=(0, 0, 1, 0.3), edgecolors='b')
    #pc_D = PolyCollection([(center, node_coords[17]) for center in edge_centers], closed=False, facecolors='none', edgecolors='r')
    # 22, 23
    pc_D = PolyCollection([(node_coords[22], node_coords[17]), (node_coords[23], node_coords[17]), (edge_centers[2], node_coords[17])], closed=False, facecolors='none', edgecolors='r')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1))

    ax.add_collection(pc_triangle)
    ax.add_collection(pc_D)

    ax.quiver(edge_centers[:, 0], edge_centers[:, 1], e1[:, 0], e1[:, 1], color='b', scale=5, width=0.015)
    ax.quiver(edge_centers[:, 0], edge_centers[:, 1], e2[:, 0], e2[:, 1], color='r', scale=5, width=0.015)

    ax.plot(triangle_coords[:, 0], triangle_coords[:, 1], 'ob')
    ax.plot(*np.column_stack((node_coords[17], node_coords[22], node_coords[23])), 'or')
    ax.plot(edge_centers[:, 0], edge_centers[:, 1], 'om')


    plt.text(node_coords[17, 0] + 0.007, node_coords[17, 1] - 0.002, r'$\bm x_j^V$')
    plt.text(node_coords[17, 0] - 0.05, node_coords[17, 1] - 0.07, r'$\Omega_j^V$')

    plt.text(triangle_coords[0, 0] + 0.007, triangle_coords[0, 1] + 0.000, r'$\bm x_{i+}^D$')
    plt.text(triangle_coords[1, 0] - 0.000, triangle_coords[1, 1] - 0.015, r'$\bm x_{i}^D$')
    plt.text(triangle_coords[2, 0] + 0.007, triangle_coords[2, 1] - 0.000, r'$\bm x_{i-}^D$')

    plt.text(edge_centers[0, 0] + 0.002, edge_centers[0, 1] + 0.01, r'$\bm x_m^*$')
    plt.text(edge_centers[1, 0] + 0.005, edge_centers[1, 1] - 0.012, r'$\bm x_{m+}^*$')
    plt.text(edge_centers[2, 0] + 0.01, edge_centers[2, 1] - 0.000, r'$\bm x_{m-}^*$')

    plt.text(node_coords[22, 0] - 0.000, node_coords[22, 1] + 0.009, r'$\bm x_{j+}^V$')
    plt.text(node_coords[23, 0] + 0.007, node_coords[23, 1] - 0.000, r'$\bm x_{j-}^V$')

    

    # plt.text(quad_cell_coords[1, 0] + 0.005, quad_cell_coords[1, 1] - 0.00, r'$\bm x_j^V$')
    # plt.text(quad_cell_coords[3, 0] - 0.005, quad_cell_coords[3, 1] - 0.012, r'$\bm x_{j+}^V$')

    # e1 = e1/20 + quad_center
    # plt.text(e1[0] + 0.01, e1[1] + 0.000, r'$\bm e_D$')
    # e2 = e2/20 + quad_center
    # plt.text(e2[0] + 0.007, e2[1] - 0.000, r'$\bm e_V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/approximation_div.pdf')
