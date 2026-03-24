import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh
import matplotlib.patches as patches
import pathlib
import sys
sys.path.append('computations')
#import utility_mvd


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.size'] = 12


# def plot(quadrangle_mesh_fname, fname):
#     nodes, quadrangles = utility_mvd.load_msh(quadrangle_mesh_fname)
#     groups, cells = utility_mvd.load_npz(pathlib.Path(quadrangle_mesh_fname).with_suffix('.npz'))
#     xmin, ymin = nodes.min(axis=0)
#     xmax, ymax = nodes.max(axis=0)

#     indexes_D = np.r_[groups[0]:groups[1], groups[2]:groups[3]]
#     indexes_V = np.r_[groups[1]:groups[2], groups[3]:groups[4]]

def plot(quadrangle_mesh, fname):
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

    cell_D = 0
    cell_V = 5
    quad_cell = 14

    quad_cell = quadrangle_nodes[quad_cell]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    pc_D = PolyCollection(cells_D_coords, facecolors='none', edgecolors='r')
    pc_V = PolyCollection(cells_V_coords, facecolors='none', edgecolors='b')

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 1, 0, 0.3), edgecolors='g')
    pc_cell_D = PolyCollection([node_coords[cell_nodes[cell_D]]], facecolors=(1, 0, 0, 0.3), edgecolors='none')
    pc_cell_V = PolyCollection([node_coords[cell_nodes[cell_V]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_cell_D)
    ax.add_collection(pc_cell_V)

    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    ax.plot(*quad_center, 'og')

    plt.text(node_coords[cell_D, 0] + 0.02, node_coords[cell_D, 1] + 0.02, r'$\bm x_i^D$')
    plt.text(node_coords[cell_D, 0] - 0.12, node_coords[cell_D, 1] + 0.04, r'$\Omega_i^D$')

    plt.text(node_coords[cell_V, 0] + 0.02, node_coords[cell_V, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[cell_V, 0] - 0.07, node_coords[cell_V, 1] + 0.08, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.02, quad_center[1] - 0.02, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.08, quad_center[1] + 0.02, r'$\Omega_m$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_color(quadrangle_mesh, fname):
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

    cell_D = 0
    cell_V = 5
    quad_cell = 14

    quad_cell = quadrangle_nodes[quad_cell]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    # quad_centers = utility.get_intersection_point_of_lines(*node_coords[quadrangle_nodes[:, ::2]].T, *node_coords[quadrangle_nodes[:, 1::2]].T)
    # quad_centers = np.array(quad_centers).T

    quadrangle_centers = utility.compute_intersection_points_v2(*node_coords[quadrangle_nodes].T)

    # фильтруем сетку вороного, чтобы получить уникальные ребра и чтобы можно было нарисовать пунктиром
    data = []
    for cell in cells_D:
        for n1, n2 in zip(cell, np.roll(cell, -1)):
            if (n1, n2) in data or (n2, n1) in data:
                continue
            else:
                data.append((n1, n2))
    
    edges_D = np.array(data)
    pc_D = PolyCollection(node_coords[edges_D], closed=False, facecolors='none', edgecolors='r')
    #pc_D = PolyCollection(cells_D_coords, facecolors='none', edgecolors='k', linestyles='--')
    pc_V = PolyCollection(cells_V_coords, facecolors='none', edgecolors='b')

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 1, 0, 0.3), edgecolors='g')
    pc_cell_D = PolyCollection([node_coords[cell_nodes[cell_D]]], facecolors=(1, 0, 0, 0.3), edgecolors='none')
    pc_cell_V = PolyCollection([node_coords[cell_nodes[cell_V]]], facecolors=(0, 0, 1, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_cell_D)
    ax.add_collection(pc_cell_V)

    # Надо убрать граничные ноды Вороного для рисунка
    node_coords_V = []
    for coords in nodes_V_coords:
        ok = True
        for coords2 in quadrangle_centers:
            if np.allclose(coords, coords2):
                ok = False
                break
        if ok:
            node_coords_V.append(coords)
    
    nodes_V_coords = np.array(node_coords_V)

    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    #ax.plot(*quad_center, 'ok', mfc='w')
    #ax.plot(*quad_center, '.k', ms=3)

    # ax.plot(*quadrangle_centers.T, '^k', mfc='w')
    # ax.plot(*quadrangle_centers.T, '.k', ms=3)
    ax.plot(*quadrangle_centers.T, 'og')

    plt.text(node_coords[cell_D, 0] + 0.02, node_coords[cell_D, 1] + 0.02, r'$\bm x_i^D$')
    plt.text(node_coords[cell_D, 0] - 0.12, node_coords[cell_D, 1] + 0.04, r'$\Omega_i^D$')

    plt.text(node_coords[cell_V, 0] + 0.02, node_coords[cell_V, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[cell_V, 0] - 0.07, node_coords[cell_V, 1] + 0.08, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.02, quad_center[1] - 0.02, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.08, quad_center[1] + 0.02, r'$\Omega_m$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def plot_bw(quadrangle_mesh, fname):
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

    cell_D = 0
    cell_V = 5
    quad_cell = 14

    quad_cell = quadrangle_nodes[quad_cell]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])

    # quad_centers = utility.get_intersection_point_of_lines(*node_coords[quadrangle_nodes[:, ::2]].T, *node_coords[quadrangle_nodes[:, 1::2]].T)
    # quad_centers = np.array(quad_centers).T

    quadrangle_centers = utility.compute_intersection_points_v2(*node_coords[quadrangle_nodes].T)

    # фильтруем сетку вороного, чтобы получить уникальные ребра и чтобы можно было нарисовать пунктиром
    data = []
    for cell in cells_D:
        for n1, n2 in zip(cell, np.roll(cell, -1)):
            if (n1, n2) in data or (n2, n1) in data:
                continue
            else:
                data.append((n1, n2))
    
    edges_D = np.array(data)
    pc_D = PolyCollection(node_coords[edges_D], closed=False, facecolors='none', edgecolors='k', linestyles='--')
    #pc_D = PolyCollection(cells_D_coords, facecolors='none', edgecolors='k', linestyles='--')
    pc_V = PolyCollection(cells_V_coords, facecolors='none', edgecolors='k')

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 0, 0, 0.3), linestyles='-.', edgecolors='k')
    pc_cell_D = PolyCollection([node_coords[cell_nodes[cell_D]]], facecolors=(0, 0, 0, 0.3), edgecolors='none')
    pc_cell_V = PolyCollection([node_coords[cell_nodes[cell_V]]], facecolors=(0, 0, 0, 0.3), edgecolors='none')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)
    ax.add_collection(pc_cell_D)
    ax.add_collection(pc_cell_V)

    # Надо убрать граничные ноды Вороного для рисунка
    node_coords_V = []
    for coords in nodes_V_coords:
        ok = True
        for coords2 in quadrangle_centers:
            if np.allclose(coords, coords2):
                ok = False
                break
        if ok:
            node_coords_V.append(coords)
    
    nodes_V_coords = np.array(node_coords_V)

    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ok', mfc='k')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'ok', mfc='w')
    #ax.plot(*quad_center, 'ok', mfc='w')
    #ax.plot(*quad_center, '.k', ms=3)

    # ax.plot(*quadrangle_centers.T, '^k', mfc='w')
    # ax.plot(*quadrangle_centers.T, '.k', ms=3)
    ax.plot(*quadrangle_centers.T, 'Xk', mfc='w')

    plt.text(node_coords[cell_D, 0] + 0.02, node_coords[cell_D, 1] + 0.02, r'$\bm x_i^D$')
    plt.text(node_coords[cell_D, 0] - 0.12, node_coords[cell_D, 1] + 0.04, r'$\Omega_i^D$')

    plt.text(node_coords[cell_V, 0] + 0.02, node_coords[cell_V, 1] + 0.01, r'$\bm x_j^V$')
    plt.text(node_coords[cell_V, 0] - 0.07, node_coords[cell_V, 1] + 0.08, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.02, quad_center[1] - 0.02, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.08, quad_center[1] + 0.02, r'$\Omega_m$')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
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

    # Надо выбрать 3 ячейки: Делоне, Вороного и четырехугольник (MVD)
    # cell_D - это ячейка для вершины Делоне, т.е. ячейка Вороного
    # Номер узла = номер его окружающей ячейки (выбор через поликолекшен фаендер в2)
    node_D = 18
    node_V = 139
    quad = 137

    quad_cell = quadrangle_nodes[quad]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])
    
    fc_D = [(1, 0, 0, 0.3) if np.array_equal(cell_nodes[node_D], cell) else 'none' for cell in cells_D]
    pc_D = PolyCollection(cells_D_coords, facecolors=fc_D, edgecolors='r')

    fc_V = [(0, 0, 1, 0.3) if np.array_equal(cell_nodes[node_V], cell) else 'none' for cell in cells_V]
    pc_V = PolyCollection(cells_V_coords, facecolors=fc_V, edgecolors='b')

    pc_quad = PolyCollection([quad_cell_coords], facecolors=(0, 1, 0, 0.3), edgecolors='g')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)

    l1 = ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    l2 = ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    l3 = ax.plot(*quad_center, 'og')

    # центр описанной окружности около треугольника
    circle_center = (node_coords[node_D] + node_coords[node_V] + quad_center) / 3

    radius = np.linalg.norm(node_coords[np.concatenate((cell_nodes[node_D], cell_nodes[node_V], quadrangle_nodes[quad]))] - circle_center, axis=1).max()
    radius *= 1.1

    circle = patches.Circle(circle_center, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc_D.set_clip_path(circle)
    pc_V.set_clip_path(circle)
    pc_quad.set_clip_path(circle)
    [o.set_clip_path(circle) for o in l1]
    [o.set_clip_path(circle) for o in l2]
    [o.set_clip_path(circle) for o in l3]

    min_x, min_y = circle_center - radius
    max_x, max_y = circle_center + radius

    plt.text(node_coords[node_D, 0] + 0.01, node_coords[node_D, 1] + 0.0075, r'$\bm x_i^D$')
    plt.text(node_coords[node_D, 0] - 0.04, node_coords[node_D, 1] - 0.02, r'$\Omega_i^D$')

    plt.text(node_coords[node_V, 0] + 0.005, node_coords[node_V, 1] - 0.0075, r'$\bm x_j^V$')
    plt.text(node_coords[node_V, 0] - 0.0075, node_coords[node_V, 1] + 0.02, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.0025, quad_center[1] + 0.0075, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.0225, quad_center[1] - 0.01375, r'$\Omega_m$')
    
    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def plot3(fname):
    quadrangle_mesh = f'meshes/rectangle/rectangle_5_quadrangle'
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

    # Надо выбрать 3 ячейки: Делоне, Вороного и четырехугольник (MVD)
    # cell_D - это ячейка для вершины Делоне, т.е. ячейка Вороного
    # Номер узла = номер его окружающей ячейки (выбор через поликолекшен фаендер в2)
    node_D = 18
    node_V = 139
    quad = 137

    quad_cell = quadrangle_nodes[quad]
    quad_cell_coords = node_coords[quad_cell]
    quad_center = utility.get_intersection_point_of_lines(*node_coords[quad_cell[::2]], *node_coords[quad_cell[1::2]])
    
    fc_D = [(1, 0, 0, 0.3) if np.array_equal(cell_nodes[node_D], cell) else 'none' for cell in cells_D]
    pc_D = PolyCollection(cells_D_coords, facecolors=fc_D, edgecolors='r')

    fc_V = [(0, 0, 1, 0.3) if np.array_equal(cell_nodes[node_V], cell) else 'none' for cell in cells_V]
    pc_V = PolyCollection(cells_V_coords, facecolors=fc_V, edgecolors='b')

    fc_quads = [(0, 1, 0, 0.3) if np.array_equal(quad_cell, quad_nodes) else 'none' for quad_nodes in quadrangle_nodes]
    pc_quad = PolyCollection(node_coords[quadrangle_nodes], facecolors=fc_quads, edgecolors='g')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    ax.add_collection(pc_D)
    ax.add_collection(pc_V)
    ax.add_collection(pc_quad)

    l1 = ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    l2 = ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    l3 = ax.plot(*quad_center, 'og')

    # центр описанной окружности около треугольника
    circle_center = (node_coords[node_D] + node_coords[node_V] + quad_center) / 3

    radius = np.linalg.norm(node_coords[np.concatenate((cell_nodes[node_D], cell_nodes[node_V], quadrangle_nodes[quad]))] - circle_center, axis=1).max()
    radius *= 1.1

    circle = patches.Circle(circle_center, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc_D.set_clip_path(circle)
    pc_V.set_clip_path(circle)
    pc_quad.set_clip_path(circle)
    [o.set_clip_path(circle) for o in l1]
    [o.set_clip_path(circle) for o in l2]
    [o.set_clip_path(circle) for o in l3]

    min_x, min_y = circle_center - radius
    max_x, max_y = circle_center + radius

    plt.text(node_coords[node_D, 0] + 0.01, node_coords[node_D, 1] + 0.0075, r'$\bm x_i^D$')
    plt.text(node_coords[node_D, 0] - 0.04, node_coords[node_D, 1] - 0.02, r'$\Omega_i^D$')

    plt.text(node_coords[node_V, 0] + 0.005, node_coords[node_V, 1] - 0.0075, r'$\bm x_j^V$')
    plt.text(node_coords[node_V, 0] - 0.0075, node_coords[node_V, 1] + 0.02, r'$\Omega_j^V$')

    plt.text(quad_center[0] + 0.0025, quad_center[1] + 0.0075, r'$\bm x_m^*$')
    plt.text(quad_center[0] - 0.0225, quad_center[1] - 0.01375, r'$\Omega_m$')
    
    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    #plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/cells.pdf')
    # plot2('images/mvd/cells_v2.pdf')
    #plot3('images/mvd/cells_v3.pdf')

    folder = 'images/unsteady_anisotropic_diffusion_reaction'
    plot_color(f'meshes/ellipse/quadrangle_{1}', f'{folder}/mesh_and_different_cells.pdf')
    plot_bw(f'meshes/ellipse/quadrangle_{1}', f'{folder}/bw/f-1.pdf')