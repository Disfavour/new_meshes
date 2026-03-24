import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh
import matplotlib.patches as patches


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.size'] = 12

def plot(quadrangle_mesh):
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


    # cell_V
    quadrangles = quadrangle_nodes[[0, 1, 2]]
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g')
    pc_V = PolyCollection([node_coords[cell_nodes[6]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)

    ax.plot(*node_coords[[3, 29, 34]].T, 'ob')
    ax.plot(*node_coords[[7, 6, 16]].T, 'or')
    ax.plot(*node_coords[36], 'Xr')

    plt.text(node_coords[6, 0] + 0.01, node_coords[6, 1] - 0.00, r'$\bm x_j^V$')
    plt.text(node_coords[6, 0] + 0.07, node_coords[6, 1] + 0.02, r'$\Omega_j^V$')

    plt.text(node_coords[29, 0] - 0.03, node_coords[29, 1] - 0.0, r'$\bm x_i^D$')
    plt.text(node_coords[34, 0] + 0.01, node_coords[34, 1] - 0.0, r'$\bm x_{i+}^D$')
    plt.text(node_coords[36, 0] + 0.00, node_coords[36, 1] - 0.025, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig('images/mvd/boundary_condition_cell_V.pdf', transparent=True)


    # cell_D
    quadrangles = quadrangle_nodes[[0, 1, 23, 25]]
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g')
    
    pc_D = PolyCollection([node_coords[cell_nodes[34].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_quad)

    ax.plot(*node_coords[[1, 3, 29, 34, 35]].T, 'ob')
    ax.plot(*node_coords[[6, 15, 16]].T, 'or')
    ax.plot(*node_coords[[36, 44]].T, 'Xr')

    plt.text(node_coords[34, 0] + 0.00, node_coords[34, 1] - 0.04, r'$\bm x_i^D$')
    plt.text(node_coords[34, 0] + 0.04, node_coords[34, 1] + 0.04, r'$\Omega_i^D$')

    plt.text(node_coords[29, 0] + 0.00, node_coords[29, 1] - 0.04, r'$\bm x_{i+}^D$')
    plt.text(node_coords[6, 0] - 0.04, node_coords[6, 1] + 0.03, r'$\bm x_j^V$')
    plt.text(node_coords[36, 0] + 0.00, node_coords[36, 1] - 0.04, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig('images/mvd/boundary_condition_cell_D.pdf', transparent=True)


    # cell_D_corner
    quadrangles = quadrangle_nodes[[0, 2, 3]]
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g')
    
    pc_D = PolyCollection([node_coords[cell_nodes[29].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_quad)

    ax.plot(*node_coords[[3, 29, 33, 34]].T, 'ob')
    ax.plot(*node_coords[[6, 7]].T, 'or')
    ax.plot(*node_coords[[36, 37]].T, 'Xr')

    plt.text(node_coords[29, 0] + 0.00, node_coords[29, 1] - 0.035, r'$\bm x_i^D$')
    plt.text(node_coords[29, 0] + 0.07, node_coords[29, 1] + 0.1, r'$\Omega_i^D$')

    plt.text(node_coords[36, 0] + 0.00, node_coords[36, 1] - 0.035, r'$\bm x_{j+}^V$')
    plt.text(node_coords[37, 0] - 0.05, node_coords[37, 1] + 0.00, r'$\bm x_j^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig('images/mvd/boundary_condition_cell_D_corner.pdf', transparent=True)
    plt.close()


def plot2(quadrangle_mesh, fname):
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

    quadrangles = quadrangle_nodes[[0, 1, 2, 23, 25]]
    
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g', linewidths=(3, 1, 1, 1, 1))
    pc_V = PolyCollection([node_coords[cell_nodes[6]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')
    pc_D = PolyCollection([node_coords[cell_nodes[34].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)

    ax.plot(*node_coords[[1, 3, 29, 34, 35]].T, 'ob')
    ax.plot(*node_coords[[6, 7, 15, 16]].T, 'or')
    ax.plot(*node_coords[[36, 44]].T, 'Xr')

    #plt.text(node_coords[6, 0] + 0.02, node_coords[6, 1] + 0.00, r'$\bm x_j^V$')
    plt.text(node_coords[6, 0] - 0.03, node_coords[6, 1] + 0.035, r'$\bm x_j^V$')
    plt.text(node_coords[6, 0] + 0.05, node_coords[6, 1] + 0.06, r'$\Omega_j^V$')
    plt.text(node_coords[34, 0] + 0.04, node_coords[34, 1] + 0.05, r'$\Omega_i^D$')

    plt.text(node_coords[34, 0] + 0.00, node_coords[34, 1] - 0.045, r'$\bm x_i^D$')
    plt.text(node_coords[29, 0] + 0.00, node_coords[29, 1] - 0.045, r'$\bm x_{i-}^D$')
    plt.text(node_coords[36, 0] + 0.00, node_coords[36, 1] - 0.045, r'$\bm x_{j-}^V$')
    #plt.text(node_coords[44, 0] + 0.00, node_coords[44, 1] - 0.045, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def plot3(quadrangle_mesh, fname):
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

    quadrangles = quadrangle_nodes[[0, 1, 2, 23, 25]]
    
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g', linewidths=(3, 1, 1, 1, 1))

    dummy_node_coords = node_coords[6].copy() # 29 6 34 - фейковый 4-х угольник
    dummy_node_coords[1] = -dummy_node_coords[1]

    dummy_quad = [node_coords[29], dummy_node_coords, node_coords[34]]
    pc_quad_dummy = PolyCollection([dummy_quad], facecolors='none', edgecolors='g', closed=False, linestyles='--', linewidths=2)

    pc_V = PolyCollection([node_coords[cell_nodes[6]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')
    pc_D = PolyCollection([node_coords[cell_nodes[34].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_quad_dummy)

    ax.plot(*node_coords[[1, 3, 29, 34, 35]].T, 'ob')
    ax.plot(*node_coords[[6, 7, 15, 16]].T, 'or')
    ax.plot(*node_coords[[36, 44]].T, 'Xr')
    ax.plot(*dummy_node_coords, 'Xr')

    #plt.text(node_coords[6, 0] + 0.02, node_coords[6, 1] + 0.00, r'$\bm x_j^V$')
    plt.text(node_coords[6, 0] - 0.03, node_coords[6, 1] + 0.035, r'$\bm x_j^V$')
    plt.text(node_coords[6, 0] + 0.05, node_coords[6, 1] + 0.06, r'$\Omega_j^V$')
    plt.text(node_coords[34, 0] + 0.04, node_coords[34, 1] + 0.05, r'$\Omega_i^D$')
    plt.text(dummy_node_coords[0] + 0.00, dummy_node_coords[1] - 0.045, r"$\bm x_{j'}^V$")

    plt.text(node_coords[34, 0] + 0.00, node_coords[34, 1] - 0.045, r'$\bm x_i^D$')
    plt.text(node_coords[29, 0] - 0.03, node_coords[29, 1] - 0.055, r'$\bm x_{i-}^D$')
    plt.text(node_coords[36, 0] + 0.00, node_coords[36, 1] - 0.045, r'$\bm x_{j-}^V$')
    #plt.text(node_coords[44, 0] + 0.00, node_coords[44, 1] - 0.045, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def plot4(quadrangle_mesh, fname):
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

    quadrangles = quadrangle_nodes[[0, 1, 2, 5, 13]]
    
    pc_quad = PolyCollection(node_coords[quadrangles], facecolors='none', edgecolors='g', linewidths=(2, 1, 1, 1, 1))

    dummy_node_coords = 2 * node_coords[20] - node_coords[2]

    dummy_quad = [node_coords[16], dummy_node_coords, node_coords[13]]
    pc_quad_dummy = PolyCollection([dummy_quad], facecolors='none', edgecolors='g', closed=False, linestyles='--')

    pc_V = PolyCollection([node_coords[cell_nodes[2]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')
    pc_D = PolyCollection([node_coords[cell_nodes[13].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    coords = np.concatenate((node_coords[[0, 1, 13, 16, 17]], node_coords[[2, 3, 4, 8]], node_coords[[20, 24]], [dummy_node_coords]))

    fig, ax = plt.subplots(figsize=utility.get_figsize(coords[:, 0].max() - coords[:, 0].min(), coords[:, 1].max() - coords[:, 1].min()), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_quad_dummy)

    ax.plot(*node_coords[[0, 1, 13, 16, 17]].T, 'ob')
    ax.plot(*node_coords[[2, 3, 4, 8]].T, 'or')
    ax.plot(*node_coords[[20, 24]].T, 'Xr')
    ax.plot(*dummy_node_coords, 'Xr')

    plt.text(node_coords[2, 0] - 0.04, node_coords[2, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[2, 0] - 0.08, node_coords[2, 1] + 0.06, r'$\Omega_j^V$')
    
    plt.text(dummy_node_coords[0] + 0.00, dummy_node_coords[1] - 0.025, r"$\bm x_{j'}^V$")

    plt.text(node_coords[13, 0] + 0.00, node_coords[13, 1] - 0.025, r'$\bm x_i^D$')
    plt.text(node_coords[13, 0] + 0.03, node_coords[13, 1] + 0.12, r'$\Omega_i^D$')

    plt.text(node_coords[16, 0] - 0.045, node_coords[16, 1] - 0.0, r'$\bm x_{i-}^D$')
    plt.text(node_coords[20, 0] + 0.00, node_coords[20, 1] - 0.035, r'$\bm x_{j-}^V$')
    #plt.text(node_coords[44, 0] + 0.00, node_coords[44, 1] - 0.045, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def plot4_bw(quadrangle_mesh, fname):
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

    quadrangles = quadrangle_nodes[[0, 1, 2, 5, 13]]

    quadrangle_centers = utility.compute_intersection_points_v2(*node_coords[quadrangles].T)
    
    # фильтруем quadrangles, чтобы получить уникальные ребра и чтобы можно было нарисовать пунктиром
    data = []
    for cell in quadrangles:
        for n1, n2 in zip(cell, np.roll(cell, -1)):
            if (n1, n2) in data or (n2, n1) in data:
                continue
            else:
                data.append((n1, n2))
    quad_edges = np.array(data)
    pc_quad = PolyCollection(node_coords[quad_edges], closed=False, facecolors='none', edgecolors='k', linewidths=1, linestyles='-.')

    dummy_node_coords = 2 * node_coords[20] - node_coords[2]

    dummy_quad = [node_coords[16], dummy_node_coords, node_coords[13]]
    pc_quad_dummy = PolyCollection([dummy_quad], facecolors='none', edgecolors='k', closed=False, linestyles=':')

    pc_V = PolyCollection([node_coords[cell_nodes[2]]], facecolors=(0, 0, 0, 0.3), edgecolors='none')
    pc_D = PolyCollection([node_coords[cell_nodes[13].astype(int)]], facecolors=(0, 0, 0, 0.3), edgecolors='none')

    coords = np.concatenate((node_coords[[0, 1, 13, 16, 17]], node_coords[[2, 3, 4, 8]], node_coords[[20, 24]], [dummy_node_coords]))

    fig, ax = plt.subplots(figsize=utility.get_figsize(coords[:, 0].max() - coords[:, 0].min(), coords[:, 1].max() - coords[:, 1].min()), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_quad_dummy)

    ax.plot(*node_coords[[0, 1, 13, 16, 17]].T, 'ok')
    # ax.plot(*node_coords[[2, 3, 4, 8, 20, 24]].T, 'ok', mfc='w')
    ax.plot(*np.concatenate((node_coords[[2, 3, 4, 8]], dummy_node_coords[np.newaxis, :])).T, 'ok', mfc='w')
    #ax.plot(*node_coords[[20, 24]].T, 'xk')
    #ax.plot(*quadrangle_centers.T, 'xk')
    #ax.plot(*dummy_node_coords, 'xk')
    ax.plot(*quadrangle_centers.T, 'Xk', mfc='w')

    plt.text(node_coords[2, 0] - 0.04, node_coords[2, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[2, 0] - 0.08, node_coords[2, 1] + 0.06, r'$\Omega_j^V$')
    
    plt.text(dummy_node_coords[0] + 0.00, dummy_node_coords[1] - 0.025, r"$\bm \tilde{\bm x}_{j}^V$")

    plt.text(node_coords[13, 0] + 0.00, node_coords[13, 1] - 0.025, r'$\bm x_i^D$')
    plt.text(node_coords[13, 0] + 0.03, node_coords[13, 1] + 0.12, r'$\Omega_i^D$')

    plt.text(node_coords[16, 0] - 0.045, node_coords[16, 1] - 0.0, r'$\bm x_{i-}^D$')
    plt.text(node_coords[20, 0] + 0.00, node_coords[20, 1] - 0.03, r'$\bm x_m^*$')
    #plt.text(node_coords[44, 0] + 0.00, node_coords[44, 1] - 0.045, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def plot4_color(quadrangle_mesh, fname):
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

    quadrangles = quadrangle_nodes[[0, 1, 2, 5, 13]]

    quadrangle_centers = utility.compute_intersection_points_v2(*node_coords[quadrangles].T)
    
    # фильтруем quadrangles, чтобы получить уникальные ребра и чтобы можно было нарисовать пунктиром
    data = []
    for cell in quadrangles:
        for n1, n2 in zip(cell, np.roll(cell, -1)):
            if (n1, n2) in data or (n2, n1) in data:
                continue
            else:
                data.append((n1, n2))
    quad_edges = np.array(data)
    pc_quad = PolyCollection(node_coords[quad_edges], closed=False, facecolors='none', edgecolors='g')

    dummy_node_coords = 2 * node_coords[20] - node_coords[2]

    dummy_quad = [node_coords[16], dummy_node_coords, node_coords[13]]
    pc_quad_dummy = PolyCollection([dummy_quad], facecolors='none', edgecolors='g', closed=False, linestyles='--')

    pc_V = PolyCollection([node_coords[cell_nodes[2]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')
    pc_D = PolyCollection([node_coords[cell_nodes[13].astype(int)]], facecolors=(1, 0, 0, 0.3), edgecolors='none')

    coords = np.concatenate((node_coords[[0, 1, 13, 16, 17]], node_coords[[2, 3, 4, 8]], node_coords[[20, 24]], [dummy_node_coords]))

    fig, ax = plt.subplots(figsize=utility.get_figsize(coords[:, 0].max() - coords[:, 0].min(), coords[:, 1].max() - coords[:, 1].min()), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_quad_dummy)

    ax.plot(*node_coords[[0, 1, 13, 16, 17]].T, 'ob')
    # ax.plot(*node_coords[[2, 3, 4, 8, 20, 24]].T, 'ok', mfc='w')
    ax.plot(*np.concatenate((node_coords[[2, 3, 4, 8]], dummy_node_coords[np.newaxis, :])).T, 'or')
    #ax.plot(*node_coords[[20, 24]].T, 'xk')
    #ax.plot(*quadrangle_centers.T, 'xk')
    #ax.plot(*dummy_node_coords, 'xk')
    ax.plot(*quadrangle_centers.T, 'og')

    plt.text(node_coords[2, 0] - 0.04, node_coords[2, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[2, 0] - 0.08, node_coords[2, 1] + 0.06, r'$\Omega_j^V$')
    
    plt.text(dummy_node_coords[0] + 0.00, dummy_node_coords[1] - 0.025, r"$\bm \tilde{\bm x}_{j}^V$")

    plt.text(node_coords[13, 0] + 0.00, node_coords[13, 1] - 0.025, r'$\bm x_i^D$')
    plt.text(node_coords[13, 0] + 0.03, node_coords[13, 1] + 0.12, r'$\Omega_i^D$')

    plt.text(node_coords[16, 0] - 0.045, node_coords[16, 1] - 0.0, r'$\bm x_{i-}^D$')
    plt.text(node_coords[20, 0] + 0.00, node_coords[20, 1] - 0.03, r'$\bm x_m^*$')
    #plt.text(node_coords[44, 0] + 0.00, node_coords[44, 1] - 0.045, r'$\bm x_{j+}^V$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


if __name__ == '__main__':
    #plot(f'meshes/rectangle/rectangle_0_quadrangle')
    #plot2(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/boundary_quadrangle_with_cells.pdf')
    #plot3(f'meshes/rectangle/rectangle_1_quadrangle', 'images/mvd/boundary_quadrangle_with_cells_v2.pdf')

    folder = 'images/unsteady_anisotropic_diffusion_reaction'
    plot4_color(f'meshes/ellipse/quadrangle_{1}', f'{folder}/boundary_conditions_approximation.pdf')
    plot4_bw(f'meshes/ellipse/quadrangle_{1}', f'{folder}/bw/f-3.pdf')
