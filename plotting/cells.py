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

    # cells_D 3, 28
    # cells_V 17
    # quad cell 10

    quad_cell = quadrangle_nodes[10]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])
    
    fc_D = [(1, 0, 0, 0.3) if np.array_equal(cell_nodes[3], cell) else 'none' for cell in cells_D]
    pc_D = PolyCollection(cells_D_coords, facecolors=fc_D, edgecolors='r')

    fc_V = [(0, 0, 1, 0.3) if np.array_equal(cell_nodes[17], cell) else 'none' for cell in cells_V]
    pc_V = PolyCollection(cells_V_coords, facecolors=fc_V, edgecolors='b')

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(1, 0, 1, 0.3), edgecolors='m')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin))

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)

    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    ax.plot(*quad_center, 'om')

    plt.text(node_coords[3, 0] + 0.02, node_coords[3, 1] - 0.00, r'$\bm x_i^D$')
    plt.text(node_coords[3, 0] - 0.12, node_coords[3, 1] - 0.05, r'$\Omega_i^D$')

    # plt.text(node_coords[28, 0] + 0.035, node_coords[27, 1] - 0.035, r'$\bm x_{i+}^D$')
    # plt.text(node_coords[28, 0] + 0.02, node_coords[27, 1] - 0.12, r'$\Omega_{i+}^D$')

    plt.text(node_coords[17, 0] + 0.013, node_coords[17, 1] - 0.04, r'$\bm x_j^V$')
    plt.text(node_coords[17, 0] - 0.06, node_coords[17, 1] - 0.08, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.015, quad_center[1] - 0.00, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.04, quad_center[1] + 0.05, r'$\Omega_m$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/cells.pdf')
