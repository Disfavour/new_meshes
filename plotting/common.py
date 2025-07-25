import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import meshio
import numpy as np


def


def get_cells_and_nodes(mesh_file):
    mesh = meshio.read(mesh_file)

    nodes = mesh.points[:, :2]
    cells = np.array([[nodes[node] for node in cell] for cell in mesh.cells[0].data])

    return cells, nodes


def plot_mesh(figsize, cell_coords, edgecolors='b'):
    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(cell_coords, closed=True, facecolors='none', edgecolors=edgecolors)
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k', linewidths=2)

    ax.add_collection(pc)
    #ax.add_collection(boundary_pc)

    ax.plot(nodes[:, 0], nodes[:, 1], 'bo')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '1.pdf'), transparent=True)
    plt.close()


def plot_mesh_from_file():
    cells, nodes = get_cells_and_nodes(triangle_mesh)


if __name__ == '__main__':
    pass