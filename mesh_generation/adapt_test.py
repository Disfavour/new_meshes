import gmsh
import sys
import math

# The API can be used to import a mesh without reading it from a file, by
# creating nodes and elements on the fly and storing them in model
# entities. These model entities can be existing CAD entities, or can be
# discrete entities, entirely defined by the mesh.
#
# Discrete entities can be reparametrized (see `t13.py') so that they can be
# remeshed later on; and they can also be combined with built-in CAD entities to
# produce hybrid models.
#
# We combine all these features in this tutorial to perform terrain meshing,
# where the terrain is described by a discrete surface (that we then
# reparametrize) combined with a CAD representation of the underground.

gmsh.initialize()

gmsh.model.add("x2")

# We will create the terrain surface mesh from N x N input data points:
N = 100


# Helper function to return a node tag given two indices i and j:
def tag(i, j):
    return (N + 1) * i + j + 1


# The x, y, z coordinates of all the nodes:
coords = []

# The tags of the corresponding nodes:
nodes = []

# The connectivities of the triangle elements (3 node tags per triangle) on the
# terrain surface:
tris = []

# The connectivities of the line elements on the 4 boundaries (2 node tags
# for each line element):
lin = [[], [], [], []]

# The connectivities of the point elements on the 4 corners (1 node tag for each
# point element):
pnt = [tag(0, 0), tag(N, 0), tag(N, N), tag(0, N)]

for i in range(N + 1):
    for j in range(N + 1):
        nodes.append(tag(i, j))
        coords.extend([
            float(i) / N,
            float(j) / N, 0.05 * math.sin(10 * float(i + j) / N)
        ])
        if i > 0 and j > 0:
            tris.extend([tag(i - 1, j - 1), tag(i, j - 1), tag(i - 1, j)])
            tris.extend([tag(i, j - 1), tag(i, j), tag(i - 1, j)])
        if (i == 0 or i == N) and j > 0:
            lin[3 if i == 0 else 1].extend([tag(i, j - 1), tag(i, j)])
        if (j == 0 or j == N) and i > 0:
            lin[0 if j == 0 else 2].extend([tag(i - 1, j), tag(i, j)])

# Create 4 discrete points for the 4 corners of the terrain surface:
for i in range(4):
    gmsh.model.addDiscreteEntity(0, i + 1)
gmsh.model.setCoordinates(1, 0, 0, coords[3 * tag(0, 0) - 1])
gmsh.model.setCoordinates(2, 1, 0, coords[3 * tag(N, 0) - 1])
gmsh.model.setCoordinates(3, 1, 1, coords[3 * tag(N, N) - 1])
gmsh.model.setCoordinates(4, 0, 1, coords[3 * tag(0, N) - 1])

# Create 4 discrete bounding curves, with their boundary points:
for i in range(4):
    gmsh.model.addDiscreteEntity(1, i + 1, [i + 1, i + 2 if i < 3 else 1])

# Create one discrete surface, with its bounding curves:
gmsh.model.addDiscreteEntity(2, 1, [1, 2, -3, -4])

# Add all the nodes on the surface (for simplicity... see below):
gmsh.model.mesh.addNodes(2, 1, nodes, coords)

# Add point elements on the 4 points, line elements on the 4 curves, and
# triangle elements on the surface:
for i in range(4):
    # Type 15 for point elements:
    gmsh.model.mesh.addElementsByType(i + 1, 15, [], [pnt[i]])
    # Type 1 for 2-node line elements:
    gmsh.model.mesh.addElementsByType(i + 1, 1, [], lin[i])
# Type 2 for 3-node triangle elements:
gmsh.model.mesh.addElementsByType(1, 2, [], tris)


gmsh.fltk.run()

gmsh.finalize()