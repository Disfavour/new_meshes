import gmsh
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import sys
sys.path.append('mesh_generation')
import utility

# ширина рабочей области страницы а4
# 21 - 3 - 1.5 = 16.5 cm
cm = 1 / 2.54
text_width = 16.5 * cm


# Для четырехугольной сетки
def plot_local_basises(mesh_name, fname):
    gmsh.initialize()
    gmsh.open(f'{mesh_name}.msh')

    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    element_node_tags = element_node_tags.reshape(-1, 4)

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)
    figsize = np.array([text_width, (ymax - ymin) * text_width / (xmax - xmin)]) / 1.6

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])
    
    gmsh.finalize()

    loaded = np.load(f'{mesh_name}.npz', allow_pickle=True)
    node_groups = loaded['node_groups']
    cells = loaded['cells']


    all_nodes = np.array(range(1, node_groups[-1] + one), dtype=np.uint64)
    D_inner_nodes = all_nodes[:node_groups[0]]
    D_boundary_nodes = all_nodes[node_groups[0]:node_groups[1]]
    V_inner_nodes = all_nodes[node_groups[1]:node_groups[2]]
    V_boundary_nodes = all_nodes[node_groups[2]:node_groups[3]]

    D_inner_nodes_coords = np.array([node_coords[node - one][:2] for node in D_inner_nodes])
    D_boundary_nodes_coords = np.array([node_coords[node - one][:2] for node in D_boundary_nodes])
    V_inner_nodes_coords = np.array([node_coords[node - one][:2] for node in V_inner_nodes])
    V_boundary_nodes_coords = np.array([node_coords[node - one][:2] for node in V_boundary_nodes])

    cell_coords = np.array([
        [node_coords[node - one][:2] for node in nodes] for nodes in element_node_tags
    ])

    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(cell_coords, closed=True, facecolors='none', edgecolors='m')
    ax.add_collection(pc)

    ax.plot(D_inner_nodes_coords[:, 0], D_inner_nodes_coords[:, 1], 'bo')
    ax.plot(D_boundary_nodes_coords[:, 0], D_boundary_nodes_coords[:, 1], 'bo', mfc='w') # mew markeredgewidth
    ax.plot(V_inner_nodes_coords[:, 0], V_inner_nodes_coords[:, 1], 'ro')
    ax.plot(V_boundary_nodes_coords[:, 0], V_boundary_nodes_coords[:, 1], 'ro', mfc='w')

    centers = [] #cell_coords.sum(axis=1) / 4
    # gmsh.get_barycenters
    basis = []
    for nodes in element_node_tags:
        centers.append(utility.get_intersection_point_of_lines(*(node_coords[node - one][:2] for node in np.concatenate((nodes[::2], nodes[1::2])))))

        e1 = (node_coords[nodes[2] - one] - node_coords[nodes[0] - one])[:2]
        e1 /= np.linalg.norm(e1)

        e2 = (node_coords[nodes[3] - one] - node_coords[nodes[1] - one])[:2]
        e2 /= np.linalg.norm(e2)

        basis.extend((e1, e2))
    
    centers = np.array(centers)
    e1 = np.array(basis[::2])
    e2 = np.array(basis[1::2])
    plt.quiver(centers[:, 0], centers[:, 1], e1[:, 0], e1[:, 1], color='b', pivot='tail')
    plt.quiver(centers[:, 0], centers[:, 1], e2[:, 0], e2[:, 1], color='r', pivot='tail')

    xmin, xmax, ymin, ymax = ax.axis('scaled')
    k = 0.03
    ax.axis([xmin - k, xmax + k, ymin - k, ymax + k])
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(f'{fname}.pdf', transparent=True)
    plt.close()


if __name__ == '__main__':
    plot_local_basises('meshes/rectangle/rectangle_0_quadrangle', 'images/mvd_meshes/local_basises')