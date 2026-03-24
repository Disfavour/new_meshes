import gmsh
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import utility


def plot(quadrangle_mesh, fname):
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
    node_groups = loaded['node_groups'].astype(int)
    cell_nodes = loaded['cells'] - 1

    nodes_D = np.concatenate((np.arange(node_groups[0]), np.arange(node_groups[1], node_groups[2])))
    nodes_D_coords = node_coords[nodes_D]

    nodes_V = np.concatenate((np.arange(node_groups[0], node_groups[1]), np.arange(node_groups[2], node_groups[3])))
    nodes_V_coords = node_coords[nodes_V]

    pc = PolyCollection(node_coords[quadrangle_nodes], facecolors='none', edgecolors='g')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin))
    
    ax.add_collection(pc)

    

    centers = []
    basis = []
    for nodes in quadrangle_nodes:
        centers.append(utility.get_intersection_point_of_lines(*(node_coords[node] for node in np.concatenate((nodes[::2], nodes[1::2])))))

        e1 = (node_coords[nodes[2]] - node_coords[nodes[0]])
        e1 /= np.linalg.norm(e1)

        e2 = (node_coords[nodes[3]] - node_coords[nodes[1]])
        e2 /= np.linalg.norm(e2)

        basis.extend((e1, e2))
    
    centers = np.array(centers)
    e1 = np.array(basis[::2])
    e2 = np.array(basis[1::2])

    #ax.plot(D_boundary_nodes_coords[:, 0], D_boundary_nodes_coords[:, 1], 'bo', mfc='w') # mew markeredgewidth
    ax.plot(centers[:, 0], centers[:, 1], 'og')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')
    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    
    
    ax.quiver(centers[:, 0], centers[:, 1], e1[:, 0], e1[:, 1], color='b', pivot='tail')
    ax.quiver(centers[:, 0], centers[:, 1], e2[:, 0], e2[:, 1], color='r', pivot='tail')

    xmin, xmax, ymin, ymax = ax.axis('scaled')
    k = 0.03
    ax.axis([xmin - k, xmax + k, ymin - k, ymax + k])
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/local_basises.pdf')