import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import utility
import gmsh


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

    pc = matplotlib.collections.PolyCollection(node_coords[quadrangle_nodes], facecolors='none', edgecolors='m')

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin))

    ax.add_collection(pc)
    ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'ob')
    ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'or')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_quadrangle', 'images/mvd/mesh_mvd.pdf')
