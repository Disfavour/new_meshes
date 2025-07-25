import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
import pickle

# rectangle_1_quadrangle.msh rectangle_1_split_quadrangles.msh rectangle_1_triangle rectangle_1_small_quadrangle
mesh = meshio.read('meshes/msh/rectangle_1_triangle.msh')

points = mesh.points[:, :2]
#quads = mesh.cells_dict['triangle']
quads = mesh.cells[0].data

quads_points = [
    np.array([points[j] for j in i])
    for i in quads
]
# print(quads_points)

# print(mesh.points)
# print(mesh.cells_dict['quadrangle'])

# 23 20 33 25
# point_numbers = [23, 20, 33, 25]
# points_boundary = points[point_numbers]
# center = points_boundary.sum(axis=0) / len(point_numbers)
# radius = np.linalg.norm(points_boundary - center, axis=1).max()
# print(center, radius)

plt.figure()
ax = plt.gca()

pc = PolyCollection(quads_points, facecolors='white', edgecolors='black')
pc.set_alpha(0.2)
#pc.set_array(np.ones(quads.shape[0]))   # scalar -> cell
ax.add_collection(pc)
ax.autoscale()
#ax.axis('equal')
ax.set_aspect('equal', 'box')

plt.scatter(points[:, 0], points[:, 1])

for i, (xi,yi) in enumerate(points):
    plt.text(xi,yi,i, size=8)

# def on_mouse_move(event):
#     print(event)
#     print(pc.contains(event))
from matplotlib.patches import Polygon

def update_polygon(polygon1):
    if polygon1 == -1:
        points = [[0, 0], [0, 0]]
    else:
        points = quads_points[polygon1]
        #print(pc.)
        #points = triang.triangles[polygon1]
    #xs = triang.x[points]
    #ys = triang.y[points]
    #polygon.set_xy(np.column_stack([xs, ys]))
    polygon.set_xy(points)


def on_mouse_move(event):
    if event.inaxes is None:
        polygon1 = -1
    else:
        if_in, items = pc.contains(event)
        polygon1 = items['ind'][0] if if_in else -1
    update_polygon(polygon1)
    ax.set_title(f'In polygon {polygon1}')
    event.canvas.draw()

fig = plt.gcf()

polygon = Polygon([[0, 0], [0, 0]], facecolor='y', alpha=0.5)  # dummy data for (xs, ys)
update_polygon(-1)
ax.add_patch(polygon)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

#ax.plot((0, 1, 2, 3))
plt.show()
