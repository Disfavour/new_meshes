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

    # quad cell 10

    quad_cell = quadrangle_nodes[10]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    e1 = quad_cell_coords[2] - quad_cell_coords[0]
    e1 /= np.linalg.norm(e1)
    e2 = quad_cell_coords[3] - quad_cell_coords[1]
    e2 /= np.linalg.norm(e2)

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(1, 0, 1, 0.3), edgecolors='m')
    pc_V = PolyCollection([quad_cell_coords[::2]], closed=False, facecolors='none', edgecolors='b')
    pc_D = PolyCollection([quad_cell_coords[1::2]], closed=False, facecolors='none', edgecolors='r')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1))

    ax.add_collection(pc_quad)
    ax.add_collection(pc_V)
    ax.add_collection(pc_D)

    ax.quiver(*quad_center, *e1, color='b', scale=5, width=0.015)
    ax.quiver(*quad_center, *e2, color='r', scale=5, width=0.015)

    ax.plot(quad_cell_coords[::2, 0], quad_cell_coords[::2, 1], 'ob')
    ax.plot(quad_cell_coords[1::2, 0], quad_cell_coords[1::2, 1], 'or')
    ax.plot(*quad_center, 'om')

    plt.text(quad_center[0] + 0.005, quad_center[1] - 0.00, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.02, quad_center[1] + 0.05, r'$\Omega_m$')

    plt.text(quad_cell_coords[0, 0] + 0.005, quad_cell_coords[0, 1] - 0.00, r'$\bm x_i^D$')
    plt.text(quad_cell_coords[2, 0] - 0.004, quad_cell_coords[2, 1] + 0.01, r'$\bm x_{i+}^D$')

    plt.text(quad_cell_coords[1, 0] + 0.005, quad_cell_coords[1, 1] - 0.00, r'$\bm x_j^V$')
    plt.text(quad_cell_coords[3, 0] - 0.005, quad_cell_coords[3, 1] - 0.012, r'$\bm x_{j+}^V$')

    e1 = e1/20 + quad_center
    plt.text(e1[0] + 0.01, e1[1] + 0.000, r'$\bm e_D$')
    e2 = e2/20 + quad_center
    plt.text(e2[0] + 0.007, e2[1] - 0.000, r'$\bm e_V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/approximation_grad.pdf')
