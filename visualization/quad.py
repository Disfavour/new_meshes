from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import matplotlib


mesh = meshio.read('meshes/msh/rectangle_1_quadrangle.msh')

l = [
    [
        [0, 0], [1, 0], [2, 0]
    ],
    [
        [0, 0.8], [1, 1.2], [2, 0.5]
    ],
    [
        [0, 2], [1, 2], [2, 2]
    ],
]
C = [
    [1, 2, 3],
    [4, 5, 6],
]
C = np.ones((3, 3))
l = np.array(l)

for m in range(2):
    for n in range(3):
        print(l[m, n])

coordinates = l

plt.figure()

qmesh = QuadMesh(l)
qmesh.set_edgecolor('r')
#qmesh.draw(renderer=plt.gcf().canvas.get_renderer()) Agg
#qmesh.draw(renderer=None)
# plt.pcolormesh(C, l)
#plt.show(renderer=plt.gcf())


print(plt.rcParams["patch.facecolor"], plt.rcParams["patch.linewidth"])

plt.pcolormesh(l[:, :, 0], l[:, :, 1], C, edgecolors='k', linewidths=4, cmap='RdBu', vmin=-1, vmax=1)
plt.show()

#print(mesh.points)

pts = mesh.points[:, :2]
print(pts)

#print(l[:, :, 0])

# x1 = np.random.randn(100);
# x2 = np.random.randn(100);
# x3 = np.random.randn(100, 100);

# fig, ax = plt.subplots();

# quadMeshCol = ax.pcolormesh(x1, x2, x3);
# plt.show()