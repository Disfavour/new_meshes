import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import utility
import gmsh


def plot(triangle_mesh, fname):
    gmsh.initialize()
    gmsh.open(triangle_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)

    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    triangle_nodes = triangle_nodes.reshape(-1, 3) - 1

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]

    gmsh.finalize()

    triangulation = matplotlib.tri.Triangulation(node_coords[:, 0], node_coords[:, 1], triangle_nodes)

    fig, ax = plt.subplots(figsize=utility.get_figsize(xmax - xmin, ymax - ymin))

    ax.triplot(triangulation, 'o-b')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    plot(f'meshes/rectangle/rectangle_0_triangle.msh', 'images/mvd/mesh_triangle.pdf')
