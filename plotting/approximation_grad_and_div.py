import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh


def plot3_bw(quadrangle_mesh, quad, fname):
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


    quad_cell = quadrangle_nodes[quad]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    e1 = quad_cell_coords[2] - quad_cell_coords[0]
    e1 /= np.linalg.norm(e1)
    e2 = quad_cell_coords[3] - quad_cell_coords[1]
    e2 /= np.linalg.norm(e2)

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 0, 0, 0.3), edgecolors='k', linestyles='-.')
    pc_V = PolyCollection([quad_cell_coords[::2]], closed=False, facecolors='none', edgecolors='k')
    pc_D = PolyCollection([quad_cell_coords[1::2]], closed=False, facecolors='none', edgecolors='k', linestyles='--')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=utility.get_default_figsize(), constrained_layout=True)

    ax1.add_collection(pc_quad)
    ax1.add_collection(pc_V)
    ax1.add_collection(pc_D)

    ax1.quiver(*quad_center, *e1, color='k', scale=5, width=0.015)
    ax1.quiver(*quad_center, *e2, color='k', scale=5, width=0.015)

    ax1.plot(quad_cell_coords[::2, 0], quad_cell_coords[::2, 1], 'ok')
    ax1.plot(quad_cell_coords[1::2, 0], quad_cell_coords[1::2, 1], 'ok', mfc='w')
    ax1.plot(*quad_center, 'Xk', mfc='w')

    ax1.text(quad_center[0] + 0.01, quad_center[1] - 0.005, r'$\bm x_m^*$')
    ax1.text(quad_center[0] - 0.07, quad_center[1] + 0.03, r'$\Omega_m$')

    ax1.text(quad_cell_coords[0, 0] + 0.007, quad_cell_coords[0, 1] + 0.00, r'$\bm x_i^D$')
    ax1.text(quad_cell_coords[2, 0] + 0.007, quad_cell_coords[2, 1] + 0.00, r'$\bm x_{i+}^D$')

    ax1.text(quad_cell_coords[1, 0] + 0.007, quad_cell_coords[1, 1] + 0.00, r'$\bm x_j^V$')
    ax1.text(quad_cell_coords[3, 0] - 0.03, quad_cell_coords[3, 1] + 0.00, r'$\bm x_{j+}^V$')

    ax1.text(quad_center[0] - 0.01, quad_center[1] + 0.03, r'$\bm e_D$')
    ax1.text(quad_center[0] - 0.03, quad_center[1] - 0.03, r'$\bm e_V$')

    ax1.axis('scaled')
    ax1.set_axis_off()

    # div
    #triangle = cell_nodes[node_V]
    node_V = 5
    triangle_coords = node_coords[cell_nodes[node_V]]

    # quad_nodes = quadrangle_nodes[quads]
    # quad_nodes_coords = node_coords[quad_nodes]

    linked_quads = []
    edge_centers = []
    for n1, n2 in zip(cell_nodes[node_V], np.roll(cell_nodes[node_V], -1)):
        linked_quads.append((((quadrangle_nodes == n1) + (quadrangle_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item())
        edge_centers.append((node_coords[n1] + node_coords[n2]) / 2)
    edge_centers = np.array(edge_centers)

    quad_nodes_coords = node_coords[quadrangle_nodes[linked_quads]]
    e1 = quad_nodes_coords[:, 2] - quad_nodes_coords[:, 0]
    e2 = quad_nodes_coords[:, 3] - quad_nodes_coords[:, 1]
    e1 /= np.linalg.norm(e1, axis=1).reshape(3, -1)
    e2 /= np.linalg.norm(e2, axis=1).reshape(3, -1)

    pc_triangle = PolyCollection([triangle_coords], facecolors=(0, 0, 0, 0.3), edgecolors='k')
    #pc_D = PolyCollection([(center, node_coords[17]) for center in edge_centers], closed=False, facecolors='none', edgecolors='r')
    # 22, 23

    pc_D = PolyCollection(np.concatenate((quad_nodes_coords[:, 1], quad_nodes_coords[:, 3]), axis=1).reshape(-1, 2, 2), closed=False, facecolors='none', edgecolors='k', linestyles='--')

    ax2.add_collection(pc_triangle)
    ax2.add_collection(pc_D)

    # ax2.quiver(edge_centers[:, 0], edge_centers[:, 1], e1[:, 0], e1[:, 1], color='k', scale=10, width=0.012)
    # ax2.quiver(edge_centers[:, 0], edge_centers[:, 1], e2[:, 0], e2[:, 1], color='k', scale=10, width=0.012)

    ax2.quiver(*edge_centers[0], *e1[0], color='k', scale=6, width=0.013)
    ax2.quiver(*edge_centers[0], *e2[0], color='k', scale=6, width=0.013)

    ax2.quiver(*edge_centers[1], *-e1[1], color='k', scale=6, width=0.013)
    ax2.quiver(*edge_centers[1], *-e2[1], color='k', scale=6, width=0.013)

    ax2.quiver(*edge_centers[2], *-e1[2], color='k', scale=6, width=0.013)
    ax2.quiver(*edge_centers[2], *-e2[2], color='k', scale=6, width=0.013)

    ax2.plot(triangle_coords[:, 0], triangle_coords[:, 1], 'ok')
    linked_quad_nodes = quadrangle_nodes[linked_quads][:, [1, 3]]
    V_nodes = np.unique(linked_quad_nodes.flat)
    V_nodes = V_nodes[:-1]
    ax2.plot(node_coords[V_nodes][:, 0], node_coords[V_nodes][:, 1], 'ok', mfc='w')
    ax2.plot(edge_centers[:, 0], edge_centers[:, 1], 'Xk', mfc='w')


    ax2.text(node_coords[node_V, 0] + 0.01, node_coords[node_V, 1] + 0.00, r'$\bm x_j^V$')
    ax2.text(node_coords[node_V, 0] - 0.05, node_coords[node_V, 1] + 0.08, r'$\Omega_j^V$')

    ax2.text(triangle_coords[0, 0] + 0.01, triangle_coords[0, 1] + 0.00, r'$\bm x_{i}^D$')
    ax2.text(triangle_coords[1, 0] + 0.01, triangle_coords[1, 1] + 0.005, r'$\bm x_{i+}^D$')
    ax2.text(triangle_coords[2, 0] + 0.01, triangle_coords[2, 1] + 0.00, r'$\bm x_{i-}^D$')

    ax2.text(edge_centers[0, 0] + 0.01, edge_centers[0, 1] + 0.005, r'$\bm x_m^*$')
    ax2.text(edge_centers[1, 0] + 0.01, edge_centers[1, 1] - 0.01, r'$\bm x_{m+}^*$')
    ax2.text(edge_centers[2, 0] + 0.01, edge_centers[2, 1] + 0.000, r'$\bm x_{m-}^*$')

    V_nodes = np.setdiff1d(V_nodes, node_V)

    ax2.text(node_coords[V_nodes[1], 0] + 0.01, node_coords[V_nodes[1], 1] + 0.00, r'$\bm x_{j+}^V$')
    ax2.text(node_coords[V_nodes[0], 0] + 0.01, node_coords[V_nodes[0], 1] - 0.01, r'$\bm x_{j-}^V$')

    ax2.axis('scaled')
    ax2.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def plot3_color(quadrangle_mesh, quad, fname):
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


    quad_cell = quadrangle_nodes[quad]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    e1 = quad_cell_coords[2] - quad_cell_coords[0]
    e1 /= np.linalg.norm(e1)
    e2 = quad_cell_coords[3] - quad_cell_coords[1]
    e2 /= np.linalg.norm(e2)

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 1, 0, 0.3), edgecolors='g')
    pc_V = PolyCollection([quad_cell_coords[::2]], closed=False, facecolors='none', edgecolors='b')
    pc_D = PolyCollection([quad_cell_coords[1::2]], closed=False, facecolors='none', edgecolors='r')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=utility.get_default_figsize(), constrained_layout=True)

    ax1.add_collection(pc_quad)
    ax1.add_collection(pc_V)
    ax1.add_collection(pc_D)

    ax1.quiver(*quad_center, *e1, color='b', scale=5, width=0.015)
    ax1.quiver(*quad_center, *e2, color='r', scale=5, width=0.015)

    ax1.plot(quad_cell_coords[::2, 0], quad_cell_coords[::2, 1], 'ob')
    ax1.plot(quad_cell_coords[1::2, 0], quad_cell_coords[1::2, 1], 'or')
    ax1.plot(*quad_center, 'og')

    ax1.text(quad_center[0] + 0.01, quad_center[1] - 0.005, r'$\bm x_m^*$')
    ax1.text(quad_center[0] - 0.07, quad_center[1] + 0.03, r'$\Omega_m$')

    ax1.text(quad_cell_coords[0, 0] + 0.007, quad_cell_coords[0, 1] + 0.00, r'$\bm x_i^D$')
    ax1.text(quad_cell_coords[2, 0] + 0.007, quad_cell_coords[2, 1] + 0.00, r'$\bm x_{i+}^D$')

    ax1.text(quad_cell_coords[1, 0] + 0.007, quad_cell_coords[1, 1] + 0.00, r'$\bm x_j^V$')
    ax1.text(quad_cell_coords[3, 0] - 0.03, quad_cell_coords[3, 1] + 0.00, r'$\bm x_{j+}^V$')

    ax1.text(quad_center[0] - 0.01, quad_center[1] + 0.03, r'$\bm e_D$')
    ax1.text(quad_center[0] - 0.03, quad_center[1] - 0.03, r'$\bm e_V$')

    ax1.axis('scaled')
    ax1.set_axis_off()

    # div
    #triangle = cell_nodes[node_V]
    node_V = 5
    triangle_coords = node_coords[cell_nodes[node_V]]

    # quad_nodes = quadrangle_nodes[quads]
    # quad_nodes_coords = node_coords[quad_nodes]

    linked_quads = []
    edge_centers = []
    for n1, n2 in zip(cell_nodes[node_V], np.roll(cell_nodes[node_V], -1)):
        linked_quads.append((((quadrangle_nodes == n1) + (quadrangle_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item())
        edge_centers.append((node_coords[n1] + node_coords[n2]) / 2)
    edge_centers = np.array(edge_centers)

    quad_nodes_coords = node_coords[quadrangle_nodes[linked_quads]]
    e1 = quad_nodes_coords[:, 2] - quad_nodes_coords[:, 0]
    e2 = quad_nodes_coords[:, 3] - quad_nodes_coords[:, 1]
    e1 /= np.linalg.norm(e1, axis=1).reshape(3, -1)
    e2 /= np.linalg.norm(e2, axis=1).reshape(3, -1)

    pc_triangle = PolyCollection([triangle_coords], facecolors=(0, 0, 1, 0.3), edgecolors='b')
    #pc_D = PolyCollection([(center, node_coords[17]) for center in edge_centers], closed=False, facecolors='none', edgecolors='r')
    # 22, 23

    pc_D = PolyCollection(np.concatenate((quad_nodes_coords[:, 1], quad_nodes_coords[:, 3]), axis=1).reshape(-1, 2, 2), closed=False, facecolors='none', edgecolors='r')

    ax2.add_collection(pc_triangle)
    ax2.add_collection(pc_D)

    # ax2.quiver(edge_centers[:, 0], edge_centers[:, 1], e1[:, 0], e1[:, 1], color='k', scale=10, width=0.012)
    # ax2.quiver(edge_centers[:, 0], edge_centers[:, 1], e2[:, 0], e2[:, 1], color='k', scale=10, width=0.012)

    ax2.quiver(*edge_centers[0], *e1[0], color='b', scale=6, width=0.013)
    ax2.quiver(*edge_centers[0], *e2[0], color='r', scale=6, width=0.013)

    ax2.quiver(*edge_centers[1], *-e1[1], color='b', scale=6, width=0.013)
    ax2.quiver(*edge_centers[1], *-e2[1], color='r', scale=6, width=0.013)

    ax2.quiver(*edge_centers[2], *-e1[2], color='b', scale=6, width=0.013)
    ax2.quiver(*edge_centers[2], *-e2[2], color='r', scale=6, width=0.013)

    ax2.plot(triangle_coords[:, 0], triangle_coords[:, 1], 'ob')
    linked_quad_nodes = quadrangle_nodes[linked_quads][:, [1, 3]]
    V_nodes = np.unique(linked_quad_nodes.flat)
    V_nodes = V_nodes[:-1]
    ax2.plot(node_coords[V_nodes][:, 0], node_coords[V_nodes][:, 1], 'or')
    ax2.plot(edge_centers[:, 0], edge_centers[:, 1], 'og')


    ax2.text(node_coords[node_V, 0] + 0.01, node_coords[node_V, 1] + 0.00, r'$\bm x_j^V$')
    ax2.text(node_coords[node_V, 0] - 0.05, node_coords[node_V, 1] + 0.08, r'$\Omega_j^V$')

    ax2.text(triangle_coords[0, 0] + 0.01, triangle_coords[0, 1] + 0.00, r'$\bm x_{i}^D$')
    ax2.text(triangle_coords[1, 0] + 0.01, triangle_coords[1, 1] + 0.005, r'$\bm x_{i+}^D$')
    ax2.text(triangle_coords[2, 0] + 0.01, triangle_coords[2, 1] + 0.00, r'$\bm x_{i-}^D$')

    ax2.text(edge_centers[0, 0] + 0.01, edge_centers[0, 1] + 0.005, r'$\bm x_m^*$')
    ax2.text(edge_centers[1, 0] + 0.01, edge_centers[1, 1] - 0.01, r'$\bm x_{m+}^*$')
    ax2.text(edge_centers[2, 0] + 0.01, edge_centers[2, 1] + 0.000, r'$\bm x_{m-}^*$')

    V_nodes = np.setdiff1d(V_nodes, node_V)

    ax2.text(node_coords[V_nodes[1], 0] + 0.01, node_coords[V_nodes[1], 1] + 0.00, r'$\bm x_{j+}^V$')
    ax2.text(node_coords[V_nodes[0], 0] + 0.01, node_coords[V_nodes[0], 1] - 0.01, r'$\bm x_{j-}^V$')

    ax2.axis('scaled')
    ax2.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    #plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/approximation_grad.pdf')
    #plot2('images/mvd/approximation_grad_v2.pdf')

    folder = 'images/unsteady_anisotropic_diffusion_reaction'
    plot3_bw(f'meshes/ellipse/quadrangle_{1}', 14, f'{folder}/bw/f-2.pdf')
    plot3_color(f'meshes/ellipse/quadrangle_{1}', 14, f'{folder}/approximation_grad_and_div.pdf')
