import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import utility
import gmsh


def mesh_with_nodes(mesh_fname, fname, node_type):
    gmsh.initialize()
    gmsh.open(mesh_fname)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)
    width = xmax - xmin
    height = ymax - ymin

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    triangle_nodes = triangle_nodes.reshape(-1, 3) - 1

    cell_edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type)
    gmsh.model.mesh.create_edges()
    cell_edges, edge_orientations = gmsh.model.mesh.get_edges(cell_edge_nodes)
    edges, edge_nodes = gmsh.model.mesh.getAllEdges()
    edge_nodes = edge_nodes.reshape(-1, 2) - 1
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    pc_cells = PolyCollection(node_coords[triangle_nodes], closed=True, facecolors='none', edgecolors='b')

    fig, ax = plt.subplots(figsize=utility.get_figsize(width, height), constrained_layout=True)

    ax.add_collection(pc_cells)

    node_format = 'ob'

    if node_type == 'node':
        ax.plot(*node_coords.T, node_format)
    elif node_type == 'center':
        triangle_centers = node_coords[triangle_nodes].sum(axis=1) / 3
        ax.plot(*triangle_centers.T, node_format)
    elif node_type == 'edge':
        edge_centers = node_coords[edge_nodes].sum(axis=1) / 2
        ax.plot(*edge_centers.T, node_format)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def centroid(triangle_node_coords):
    return triangle_node_coords.sum(axis=1) / 3


def circumcenter(triangle_node_coords):
    Ax, Ay = triangle_node_coords[:, 0].T
    Bx, By = triangle_node_coords[:, 1].T
    Cx, Cy = triangle_node_coords[:, 2].T
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D
    return np.stack((Ux, Uy), axis=1)


def incenter(triangle_node_coords):
    edge_lenghts = np.linalg.norm(np.roll(triangle_node_coords, -1, axis=1) - triangle_node_coords, axis=2)
    return np.sum(triangle_node_coords * np.roll(edge_lenghts, -1, axis=1)[:, :, np.newaxis], axis=1) / edge_lenghts.sum(axis=1)[:, np.newaxis]


def orthocenter(triangle_node_coords):
    ab = triangle_node_coords[:, 1:] - triangle_node_coords[:, :2]
    # a = ab[:, 0], b = ab[:, 1]
    c = np.sum(triangle_node_coords[:, 2::-2] * ab, axis=2)
    return np.linalg.solve(ab, c)


def mesh_with_different_centers(mesh_fname, fname):
    gmsh.initialize()
    gmsh.open(mesh_fname)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)
    width = xmax - xmin
    height = ymax - ymin

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    triangle_nodes = triangle_nodes.reshape(-1, 3) - 1

    cell_edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type)
    gmsh.model.mesh.create_edges()
    cell_edges, edge_orientations = gmsh.model.mesh.get_edges(cell_edge_nodes)
    edges, edge_nodes = gmsh.model.mesh.getAllEdges()
    edge_nodes = edge_nodes.reshape(-1, 2) - 1
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    pc_cells = PolyCollection(node_coords[triangle_nodes], closed=True, facecolors='none', edgecolors='b')

    fig, ax = plt.subplots(figsize=utility.get_figsize(width, height), constrained_layout=True)

    ax.add_collection(pc_cells)

    triangle_node_coords = node_coords[triangle_nodes]
    
    ax.plot(*centroid(triangle_node_coords).T, 'o', ms=10, label='centroid')
    ax.plot(*circumcenter(triangle_node_coords).T, 'o', ms=8, label='circumcenter')
    ax.plot(*incenter(triangle_node_coords).T, 'o', ms=6, label='incenter')
    ax.plot(*orthocenter(triangle_node_coords).T, 'o', ms=4, label='orthocenter')

    fig.legend()

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)
    plt.close()


def different_triangle_centers(fname):
    triangle_node_coords1 = np.array(((
        (0, 0),
        (1, 0),
        (0.5, np.sqrt(3)/2),
    ),))

    triangle_node_coords2 = np.array(((
        (0, 0),
        (1, 0),
        (0.6, np.sqrt(3)/4),
    ),))

    fig, axs = plt.subplots(1, 2, figsize=utility.get_default_figsize(), constrained_layout=True)

    artists = []

    for ax, triangle_node_coords in zip(axs, (triangle_node_coords1, triangle_node_coords2)):
        pc_cells = PolyCollection(triangle_node_coords, closed=True, facecolors='none', edgecolors='b')

        ax.add_collection(pc_cells)
        ms = 13
        delta = 3
        artists.append(ax.plot(*centroid(triangle_node_coords).T, 'o', ms=ms))
        ms -= delta
        artists.append(ax.plot(*circumcenter(triangle_node_coords).T, 'o', ms=ms))
        ms -= delta
        artists.append(ax.plot(*incenter(triangle_node_coords).T, 'o', ms=ms))
        ms -= delta
        artists.append(ax.plot(*orthocenter(triangle_node_coords).T, 'o', ms=ms))

        ax.axis('scaled')
        ax.set_axis_off()
    

    artists = [a[0] for a in artists]
    fig.legend(artists, ('centroid', 'circumcenter', 'incenter', 'orthocenter'), loc='upper center')

    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    mesh_with_nodes(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/node_approximation.pdf', 'node')
    mesh_with_nodes(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/edge_approximation.pdf', 'edge')
    mesh_with_nodes(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/center_approximation.pdf', 'center')

    mesh_with_different_centers(f'meshes/rectangle/rectangle_1_triangle.msh', 'images/fvm_edge/different_center_approximations.pdf')

    different_triangle_centers(f'images/fvm_edge/different_triangle_centers.pdf')

    #folder = 'images/unsteady_anisotropic_diffusion_reaction'
