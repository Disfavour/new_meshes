import gmsh
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import utility


# 2D
def get_intersection_point_of_lines(a1, a2, b1, b2):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return p_x, p_y


# Для четырехугольной сетки
def plot_local_basises(quadrangle_mesh, fname):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    plt.rcParams['font.size'] = 12

    gmsh.initialize()
    gmsh.open(f'{quadrangle_mesh}.msh')

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)

    quadrangle_tags, quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    quadrangle_nodes = quadrangle_nodes.reshape(-1, 4) - 1

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    loaded = np.load(f'{quadrangle_mesh}.npz', allow_pickle=True)
    node_groups = loaded['node_groups']
    cell_nodes = loaded['cells'] - 1

    voronoi_cells = np.concatenate((cell_nodes[:node_groups[0]], cell_nodes[node_groups[1]:node_groups[2]]))
    voronoi_nodes = np.unique(np.concatenate(voronoi_cells, axis=None))
    voronoi_nodes_coords = node_coords[voronoi_nodes]
    voronoi_cells_coords = [node_coords[cell] for cell in voronoi_cells]

    triangles = np.concatenate((cell_nodes[node_groups[0]:node_groups[1]], cell_nodes[node_groups[2]]))
    triangles_nodes = np.unique(np.concatenate(triangles, axis=None))
    triangles_nodes_coords = node_coords[triangles_nodes]

    fc = [(1, 0, 1, 0.3) if np.array_equal(quadrangle_nodes[32], current_cell_nodes) else 'none' for current_cell_nodes in quadrangle_nodes]
    pc = PolyCollection(node_coords[quadrangle_nodes], facecolors=fc, edgecolors='m')

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1))
    
    ax.add_collection(pc)

    #ax.plot(D_boundary_nodes_coords[:, 0], D_boundary_nodes_coords[:, 1], 'bo', mfc='w') # mew markeredgewidth
    lines1 = ax.plot(voronoi_nodes_coords[:, 0], voronoi_nodes_coords[:, 1], 'or')
    lines2 = ax.plot(triangles_nodes_coords[:, 0], triangles_nodes_coords[:, 1], 'ob')

    centers = [] #cell_coords.sum(axis=1) / 4
    # gmsh.get_barycenters
    basis = []
    for nodes in quadrangle_nodes:
        centers.append(get_intersection_point_of_lines(*(node_coords[node] for node in np.concatenate((nodes[::2], nodes[1::2])))))

        e1 = (node_coords[nodes[2]] - node_coords[nodes[0]])
        e1 /= np.linalg.norm(e1)

        e2 = (node_coords[nodes[3]] - node_coords[nodes[1]])
        e2 /= np.linalg.norm(e2)

        basis.extend((e1, e2))
    
    centers = np.array(centers)
    e1 = np.array(basis[::2])
    e2 = np.array(basis[1::2])
    plt.quiver(centers[32, 0], centers[32, 1], e1[32, 0], e1[32, 1], color='b', pivot='tail', scale=5, width=0.015)
    plt.quiver(centers[32, 0], centers[32, 1], e2[32, 0], e2[32, 1], color='r', pivot='tail', scale=5, width=0.015)

    plt.text(centers[32, 0] - 0.007, centers[32, 1] - 0.013, r'$\bm x_m^*$')

    plt.text(centers[32, 0] - 0.04, centers[32, 1] - 0.04, r'$\Omega_m$')

    x_i_D = node_coords[quadrangle_nodes[32, 0]]
    plt.text(x_i_D[0] + 0.01, x_i_D[1], r'$\bm x_i^D$')

    x_j_V = node_coords[quadrangle_nodes[32, 1]]
    plt.text(x_j_V[0] - 0.007, x_j_V[1] + 0.011, r'$\bm x_j^V$')

    r = 0.15
    center = centers[32]
    circle = patches.Circle(center, radius=r, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in lines1]
    [o.set_clip_path(circle) for o in lines2]

    min_x, min_y = center - r
    max_x, max_y = center + r

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #xmin, xmax, ymin, ymax = ax.axis('scaled')
    #k = 0.03
    #ax.axis([xmin - k, xmax + k, ymin - k, ymax + k])
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    # 26 32
    plot_local_basises(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/cell_for_vector_function.pdf')