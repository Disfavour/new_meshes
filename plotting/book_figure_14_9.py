import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gmsh
import os

gmsh.initialize()

triangle_mesh = os.path.join('meshes', 'msh', 'rectangle_1_triangle.msh')
gmsh.open(triangle_mesh)

triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
node_coords = node_coords.reshape(-1, 3)
one = np.uint64(1)

triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
triangle_nodes = triangle_nodes.reshape(-1, 3)
gmsh.finalize()

ax = plt.figure(figsize=[6.4, 6.4]).add_subplot(projection='3d')

verts = [
    [node_coords[node - one] for node in nodes] for nodes in triangle_nodes
]

# 11    12, 10, 11  10, 13, 11  13, 6, 11   6, 5, 11    5, 12, 11
# 0     14, 4, 0    9, 14, 0    9 14 4
z = 1
node_coords = node_coords.copy()
node_coords[11][2] = z
triangle_nodes_11 = [
    [12, 10, 11],
    [10, 13, 11],
    [13, 6, 11],
    [6, 5, 11],
    [5, 12, 11],
]
verts_11 = [
    [node_coords[node] for node in nodes] for nodes in triangle_nodes_11
]
#verts = verts_11 + verts

zero_coords = node_coords[0].copy()
zero_coords[2] = 0
#print(zero_coords)

node_coords[0][2] = z
triangle_nodes_0 = [
    [14, 4, 0],
    [9, 14, 0],
    [4, 0, 0],
    [9, 0, 0],
]

verts_0 = [
    [node_coords[node] for node in nodes] for nodes in triangle_nodes_0
]

verts_0[2][2] = zero_coords
verts_0[3][2] = zero_coords

for i in verts_0:
    print(i)

verts = verts_0 + verts_11 + verts

edgecolors = [(1, 0, 0, 1) for i in range(9)] + [(0, 0, 1, 1) for i in range(triangle_nodes.shape[0])]
facecolors = [(1, 0, 0, 0.4) for i in range(9)] + [(0, 0, 0, 0) for i in range(triangle_nodes.shape[0])]
poly = Poly3DCollection(verts, edgecolors=edgecolors, facecolors=facecolors, lw=2, zorder=[1 for i in range(triangle_nodes.shape[0])] + [2 for i in range(7)])#, alpha=.7)
ax.add_collection3d(poly)

ax.view_init(elev=30, azim=-50, roll=0)
#ax.set_aspect('equalxy')

ax.set_xlim3d(0, 1)
ax.set_zlim3d(0, 1.5*z)
ax.set_ylim3d(0, 0.75)

#ax.axis('scaled')
ax.set_axis_off()

plt.tight_layout(pad=0)
plt.savefig(os.path.join('images', 'fig_14_9-2.pdf'), transparent=True)

plt.show()