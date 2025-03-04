import matplotlib.pyplot as plt 
import matplotlib.tri as tri
import matplotlib.patches as patches
import meshio


mesh = meshio.read('meshes/triangle_1.msh')

# print(mesh.cells[0].data)
# print(mesh.cells[0].type)
# print(len(mesh.cells))
# print(mesh.points)
# print(mesh.cells_dict['triangle'])

x, y = mesh.points[:, 0], mesh.points[:, 1]
triangles = mesh.cells_dict['triangle']

triangulation = tri.Triangulation(x, y, triangles) # tri.Triangulation(x, y)

trifinder = triangulation.get_trifinder()

print(trifinder(0.1, 0.5))
print(triangulation.triangles)

plt.triplot(triangulation, '-o')

polygon = patches.Polygon([[0, 0], [0, 0.1], [0.1, 0.1]], facecolor='y')
plt.gca().add_patch(polygon)

plt.show()
