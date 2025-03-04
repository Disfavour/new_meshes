import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
import pickle
from matplotlib.patches import Polygon


mesh = meshio.read('meshes/msh/rectangle_1_quadrangle.msh')

points = mesh.points[:, :2]
quads = mesh.cells_dict['quad']

quads_points = [
    np.array([points[j] for j in i])
    for i in quads
]
print(quads_points)

print(mesh.points)
print(mesh.cells_dict['quad'])

plt.figure()
ax = plt.gca()

with open('data.txt', 'rb') as convert_file:
    delaunay_volume = pickle.load(convert_file)
    voronoi_volume = pickle.load(convert_file)

delaunay_volume_d = {key: [points[int(value - 1)] for value in values] for key, values in delaunay_volume.items()}
voronoi_volume_d = {key: [points[int(value - 1)] for value in values] for key, values in voronoi_volume.items()}

triangles = [
    [points[int(node - 1)] for node in value] for key, value in voronoi_volume.items()
]

voronoi_shapes = [
    [points[int(node - 1)] for node in value] for key, value in delaunay_volume.items()
]

pc_delaunay = PolyCollection(triangles, facecolors='white', edgecolors='blue', lw=2)
pc_delaunay.set_alpha(0.5)
ax.add_collection(pc_delaunay)

pc_voronoi = PolyCollection(voronoi_shapes, facecolors='white', edgecolors='red', lw=2)
pc_voronoi.set_alpha(0.5)
ax.add_collection(pc_voronoi)



ax.autoscale()
#ax.axis('equal')
ax.set_aspect('equal', 'box')



plt.scatter(points[:, 0], points[:, 1])


for i, (xi,yi) in enumerate(points, 1):
    plt.text(xi,yi,i, size=8)




def update_polygon(point):
    points1 = [[0, 0], [0, 0]]
    polygon_delaunay.set_xy(points1)
    polygon_voronoi.set_xy(points1)

    name = 'None'

    if point >= 0:
        if point in delaunay_volume:
            points1 = delaunay_volume_d[point]
            polygon_voronoi.set_xy(points1)
            #print(point, 'del', delaunay_volume[point])
            name = f'Delaunay point {point}'
            
        elif point in voronoi_volume:
            points1 = voronoi_volume_d[point]
            polygon_delaunay.set_xy(points1)
            name = f'Voronoi point {point}'
        
            #print(point, 'vor', voronoi_volume[point])
            
        return name

nodes = np.array(points)


def get_closest_point(event):
    x, y = event.xdata, event.ydata

    vec = np.array((x, y)) - nodes
    distances = vec[:, 0] ** 2 + vec[:, 1] ** 2

    closest_point = np.argmin(distances)

    return closest_point + 1    # у нас все с единицы 

def on_mouse_move(event):
    if event.inaxes is None:
        point = -1
    else:
        if_in, items = pc_delaunay.contains(event)
        if if_in:
            point = get_closest_point(event)
        else:
            point = -1
    
    name = update_polygon(point)

        #print(event.xdata, event.ydata)
        #polygon1 = items['ind'][0] if if_in else -1
    #update_polygon(polygon1)
    ax.set_title(name)
    event.canvas.draw()
    #print(event)



fig = plt.gcf()

polygon_delaunay = Polygon([[0, 0], [0, 0]], facecolor='blue', alpha=0.3)
#update_polygon(-1)
ax.add_patch(polygon_delaunay)

polygon_voronoi = Polygon([[0, 0], [0, 0]], facecolor='red', alpha=0.3)
#update_polygon(-1)
ax.add_patch(polygon_voronoi)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

#ax.plot((0, 1, 2, 3))
plt.show()
