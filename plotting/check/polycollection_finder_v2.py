import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
import pickle
import gmsh

# rectangle_1_quadrangle.msh rectangle_1_split_quadrangles.msh rectangle_1_triangle rectangle_1_small_quadrangle
n = 1
quadrangle_mesh = f'meshes/ellipse/quadrangle_{1}'

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

nodes_D = np.concatenate((np.arange(node_groups[0]), np.arange(node_groups[1], node_groups[2])))
nodes_D_coords = node_coords[nodes_D]

nodes_V = np.concatenate((np.arange(node_groups[0], node_groups[1]), np.arange(node_groups[2], node_groups[3])))
nodes_V_coords = node_coords[nodes_V]

plt.figure()
ax = plt.gca()

for node_tag, node_coord in zip(node_tags - 1, node_coords):
    plt.text(*node_coord, node_tag, size=8)

# for tag, coord in zip(nodes_D, nodes_D_coords):
#     plt.text(*coord, tag, size=8)

# for tag, coord in zip(nodes_V, nodes_V_coords):
#     plt.text(*coord, tag, size=8)
    
quads_points = node_coords[quadrangle_nodes]
pc = PolyCollection(quads_points, facecolors='white', edgecolors='black')
pc.set_alpha(0.2)
#pc.set_array(np.ones(quads.shape[0]))   # scalar -> cell
ax.add_collection(pc)
ax.autoscale()
#ax.axis('equal')
ax.set_aspect('equal', 'box')

plt.scatter(nodes_D_coords[:, 0], nodes_D_coords[:, 1], c='b')
plt.scatter(nodes_V_coords[:, 0], nodes_V_coords[:, 1], c='r')



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
