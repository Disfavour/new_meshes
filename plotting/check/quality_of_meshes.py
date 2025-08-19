import gmsh
import matplotlib.pyplot as plt
import numpy as np

qualities = []

gmsh.initialize()
for i in range(10):
    gmsh.open(f'meshes/rectangle/rectangle_{i}_quadrangle.msh')

    quadrangle_tags, quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    quadrangle_nodes = quadrangle_nodes.reshape(-1, 4) - 1

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    q = gmsh.model.mesh.get_element_qualities(quadrangle_tags, 'gamma')
    qualities.append((q.min(), q.max(), q.mean()))
gmsh.finalize()

qualities = np.array(qualities)
x = np.arange(10)

plt.plot(x, qualities)
plt.legend(('min', 'max', 'mean'))

plt.show()