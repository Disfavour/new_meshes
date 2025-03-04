import shapely
import meshio
import matplotlib.pyplot as plt
import shapely.ops
from shapely.plotting import plot_polygon, plot_points, plot_line
import numpy as np


mesh = meshio.read('meshes/quad_1.msh')

points = mesh.points[:, :2]
quads = mesh.cells_dict['quad']

points = shapely.MultiPoint(points)#[shapely.Point(*p) for p in points]

triangles = shapely.ops.triangulate(points)

fig = plt.figure(1, dpi=90)

ax = fig.add_subplot(111)

for triangle in triangles:
    plot_polygon(triangle, ax=ax, add_points=False, color='blue')

plot_points(points, ax=ax, color='gray')

#plt.set_limits(ax, -1, 4, -1, 3)
# plt.xlim(-1, 4)
# plt.ylim(-1, 3)

plt.show()
#plt.figure()

regions = shapely.ops.voronoi_diagram(points)

fig = plt.figure(1, dpi=90)

ax = fig.add_subplot(111)

for region in regions.geoms:
    plot_polygon(region, ax=ax, add_points=False, color='blue')

plot_points(points, ax=ax, color='gray')

#set_limits(ax, -1, 4, -1, 3)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()

mesh = meshio.read('meshes/quad_1.msh')

points = mesh.points[:, :2]
quads = mesh.cells_dict['quad']

quads_points = [
    np.array([points[j] for j in i])
    for i in quads
]

rings = [shapely.LinearRing(p) for p in quads_points]

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(111)

for ring in rings:
    plot_line(ring, ax=ax, add_points=False, color='BLUE', alpha=0.7)
    plot_points(ring, ax=ax, color='gray', alpha=0.7)

plt.show()

# Polygon

polygons = [shapely.Polygon(p) for p in quads_points]

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(111)

for polygon in polygons:
    plot_polygon(polygon, ax=ax, add_points=False, color='BLUE')
    plot_points(polygon, ax=ax, color='gray', alpha=0.7)

plt.show()