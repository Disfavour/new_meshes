import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
import pickle
from triangle_circumcenter import check_angles1


def show_mesh(fname, figname, fsize):
    mesh = meshio.read(fname)

    points = mesh.points[:, :2]

    quads = mesh.cells_dict['quad'] if 'quad' in mesh.cells_dict else mesh.cells_dict['triangle']

    quads_points = [
        np.array([points[j] for j in i])
        for i in quads
    ]

    plt.figure(figsize=fsize, dpi=300, tight_layout=True)
    ax = plt.gca()

    #plt.suptitle(fname)
    min_angle = 500
    max_angle = 0
    for element in quads:
        angles = check_angles1(map(lambda tag: points[tag], element))

        min_angle = min((min_angle, *angles))
        max_angle = max((max_angle, *angles))
    
    print(fname)
    print(f'Nodes\t\t{points.shape[0]:12}')
    print(f'Cells\t\t{quads.shape[0]:12}')
    print(f'Min angle\t{min_angle:12.2f}')
    print(f'Max angle\t{max_angle:12.2f}')

    #print(f'{fname} number of nodes {points.shape[0]} number of cells {quads.shape[0]} min angle {min_angle:.2f} max angle')

    pc = PolyCollection(quads_points, facecolors='white', edgecolors='black')
    #pc.set_alpha(0)
    #pc.set_array(np.ones(quads.shape[0]))   # scalar -> cell
    ax.add_collection(pc)
    ax.autoscale()
    #ax.axis('equal')
    ax.set_aspect('equal', 'box')

    #plt.scatter(points[:, 0], points[:, 1])

    ax.set_axis_off()

    #ax.plot((0, 1, 2, 3))
    #plt.show()
    plt.savefig(figname, transparent=True)


if __name__ == '__main__':
    pass
