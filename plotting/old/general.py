import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import sys
sys.path.append('mesh_generation')
import utility


import meshio
def get_nodes_and_cells(mesh_file):
    gmsh.initialize()

    gmsh.open(mesh_file)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    one = np.uint64(1)

    element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()
    cells = np.array([node_coords[node - one] for node in node_tags[0]])

    if element_types[0] == gmsh.model.mesh.get_element_type("Triangle", 1):
        cells = cells.reshape(-1, 3)
    elif element_types[0] == gmsh.model.mesh.get_element_type("Quadrangle", 1):
        cells = cells.reshape(-1, 4)
    else:
        raise Exception('Element type is unclear')

    gmsh.finalize()

    # mesh = meshio.read(meshmesh_filename)

    # node_coords = mesh.points[:, :2]
    # cells = np.array([[node_coords[node] for node in cell] for cell in mesh.cells[0].data])

    return node_coords, cells


def Delaunay_and_Voronoi(triangle_mesh):
    Delaunay, Voronoi = [], []
    Delaunay_points, Voronoi_points = [], []
    # ключ словары - инт, иначе проблемы
    Voronoi_node_to_Delaunay_cell = {}
    Delaunay_node_to_Voronoi_cell = {}

    gmsh.initialize()

    gmsh.open(triangle_mesh)

    one = np.uint64(1)
    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    Delaunay_points = node_coords[:, :2]

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            boundaries = gmsh.model.get_boundary(((dim, surf_tag),), oriented=False)
        
            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
            Delaunay.extend([
                [node_coords[node - one][:2] for node in current_triangle_nodes] for current_triangle_nodes in triangle_nodes.reshape(-1, 3)
            ])
            
            circumcenter_coords = []
            for nodes in triangle_nodes.reshape((-1, 3)):
                circumcenter_coord = utility.circumcenter(*(node_coords[node - one] for node in nodes))
                circumcenter_coords.append(circumcenter_coord[:2])

            triangle_to_circumcenter_coords = {triangle: circumcenter_coord for triangle, circumcenter_coord in zip(triangle_tags, circumcenter_coords)}
            Voronoi_points = circumcenter_coords

            v_to_d = {}
            circumcenter_to_triangle_tag = utility.reverse_dict(triangle_to_circumcenter_coords)
            Voronoi_node_to_Delaunay_cell.update({tuple(circumcenter): [node_coords[node - one][:2] for node in nodes] for circumcenter, nodes in zip(circumcenter_coords, triangle_nodes.reshape(-1, 3))})

            edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)
            edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
            triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
            edge_to_triangles = utility.reverse_dict(triangle_to_edges)

            for edge_tag in edge_tags:
                linked_triangles = edge_to_triangles[edge_tag]

                if len(linked_triangles) == 2:
                    Voronoi.append((triangle_to_circumcenter_coords[linked_triangles[0]], triangle_to_circumcenter_coords[linked_triangles[1]]))
                else:
                    edge_nodes = edge_to_nodes[edge_tag]
                    edge_nodes_coords = [node_coords[node - one][:2] for node in edge_nodes]
                    Voronoi.append((triangle_to_circumcenter_coords[linked_triangles[0]], (edge_nodes_coords[0] + edge_nodes_coords[1]) / 2))
                    Voronoi_points.append((edge_nodes_coords[0] + edge_nodes_coords[1]) / 2)

    gmsh.finalize()
    return map(np.array, (Delaunay, Voronoi, Delaunay_points, Voronoi_points))#, Voronoi_node_to_Delaunay_cell


def quad_cells(triangle_mesh, quadrangle_mesh, small_quadrangle_mesh, fname, figsize=(6.4, 3.6), marker_size=5, dpi=300):
    Delaunay, Voronoi, Delaunay_points, Voronoi_points = Delaunay_and_Voronoi(triangle_mesh)

    gmsh.initialize()

    gmsh.open(quadrangle_mesh)
    
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    quadrangle_tags, quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(quadrangle_type)
    quadrangles = [[node_coords[node - one][:2] for node in current_quadrangle_nodes] for current_quadrangle_nodes in quadrangle_nodes.reshape(-1, 4)]
    quadrangle_centroids = gmsh.model.mesh.get_barycenters(quadrangle_type, -1, fast=False, primary=True).reshape(-1, 3)[:, :2]


    gmsh.open(small_quadrangle_mesh)
    
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)

    small_quadrangle_tags, small_quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(quadrangle_type)
    small_quadrangles = [[node_coords[node - one][:2] for node in current_small_quadrangle_nodes] for current_small_quadrangle_nodes in small_quadrangle_nodes.reshape(-1, 4)]
    small_quadrangle_centroids = gmsh.model.mesh.get_barycenters(quadrangle_type, -1, fast=False, primary=True).reshape(-1, 3)[:, :2]

    gmsh.finalize()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    d = PolyCollection(Delaunay, closed=True, facecolors='none', edgecolors='blue', zorder=1)
    v = PolyCollection(Voronoi, closed=False, facecolors='none', edgecolors='red', linestyles='--', zorder=2)

    ax.add_collection(d)
    ax.add_collection(v)

    ax.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_points[:, 0], Voronoi_points[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()


    def get_closest_point(point, points):
        vec = point - points
        distances = vec[:, 0] ** 2 + vec[:, 1] ** 2
        closest_point = np.argmin(distances)
        closest_distance = distances.min()
        return closest_point, closest_distance

    def update_polygon(point):
        if point is None:
            points = [[0, 0], [0, 0]]
        else:
            quadrangle_closesest_point, quadrangle_closest_distance = get_closest_point(point, quadrangle_centroids)
            small_quadrangle_closesest_point, small_quadrangle_closest_distance = get_closest_point(point, small_quadrangle_centroids)

            closest_cell, cells = None, None
            if quadrangle_closest_distance <= small_quadrangle_closest_distance:
                closest_cell, cells = quadrangle_closesest_point, quadrangles
            else:
                closest_cell, cells = small_quadrangle_closesest_point, small_quadrangles
            
            points = cells[closest_cell]

        polygon.set_xy(points)


    def on_mouse_move(event):
        if event.inaxes is None:
            point = None
        else:
            if_in, items = d.contains(event)
            if if_in:
                point = np.array((event.xdata, event.ydata))
            else:
                point = None
                    
        update_polygon(point)
        event.canvas.draw()

    def create_polygon(point):
        if point is None:
            return
        
        quadrangle_closesest_point, quadrangle_closest_distance = get_closest_point(point, quadrangle_centroids)
        small_quadrangle_closesest_point, small_quadrangle_closest_distance = get_closest_point(point, small_quadrangle_centroids)

        closest_cell, cells = None, None
        if quadrangle_closest_distance <= small_quadrangle_closest_distance:
            ax.add_patch(Polygon(quadrangles[quadrangle_closesest_point], facecolor='m', alpha=0.5))
        else:
            ax.add_patch(Polygon(small_quadrangles[small_quadrangle_closesest_point], facecolor='g', alpha=0.5))


    def on_mouse_click(event):
        if event.inaxes is None:
            point = None
        else:
            if_in, items = d.contains(event)
            if if_in:
                point = np.array((event.xdata, event.ydata))
            else:
                point = None
                    
        create_polygon(point)
        event.canvas.draw()

    polygon = Polygon([[0, 0], [0, 0]], facecolor='y', alpha=0.3)
    update_polygon(-1)
    ax.add_patch(polygon)

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    def save_fig(event):
        if fname is not None:
            fig.savefig(fname, transparent=True)

    fig.canvas.mpl_connect('axes_leave_event', save_fig)

    plt.show()


def approximation_cells(triangle_mesh, quadrangle_mesh, small_quadrangle_mesh, fname, figsize=(6.4, 3.6), marker_size=5, dpi=300):
    Delaunay, Voronoi, Delaunay_points, Voronoi_points, Voronoi_node_to_Delaunay_cell = Delaunay_and_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    d = PolyCollection(Delaunay, closed=True, facecolors='white', edgecolors='blue')
    v = PolyCollection(Voronoi, closed=False, facecolors='white', edgecolors='red', linestyles='--')

    ax.add_collection(d)
    ax.add_collection(v)

    ax.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_points[:, 0], Voronoi_points[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()


    def get_closest_point(point, points):
        vec = point - points
        distances = vec[:, 0] ** 2 + vec[:, 1] ** 2
        closest_point = np.argmin(distances)
        return closest_point

    def update_polygon(point):
        if point is None:
            points = [[0, 0], [0, 0]]
        else:
            closesest_point = get_closest_point(point, Voronoi_points)
            points = Voronoi_node_to_Delaunay_cell[tuple(Voronoi_points[closesest_point])]

        polygon.set_xy(points)


    def on_mouse_move(event):
        if event.inaxes is None:
            point = None
        else:
            if_in, items = d.contains(event)
            if if_in:
                point = np.array((event.xdata, event.ydata))
            else:
                point = None
        
        update_polygon(point)
        event.canvas.draw()

    def create_polygon(point):
        if point is None:
            return
        
        quadrangle_closesest_point, quadrangle_closest_distance = get_closest_point(point, quadrangle_centroids)
        small_quadrangle_closesest_point, small_quadrangle_closest_distance = get_closest_point(point, small_quadrangle_centroids)

        closest_cell, cells = None, None
        if quadrangle_closest_distance <= small_quadrangle_closest_distance:
            ax.add_patch(Polygon(quadrangles[quadrangle_closesest_point], facecolor='m', alpha=0.5))
        else:
            ax.add_patch(Polygon(small_quadrangles[small_quadrangle_closesest_point], facecolor='g', alpha=0.5))


    def on_mouse_click(event):
        if event.inaxes is None:
            point = None
        else:
            if_in, items = d.contains(event)
            if if_in:
                point = np.array((event.xdata, event.ydata))
            else:
                point = None
        
        create_polygon(point)
        event.canvas.draw()

    polygon = Polygon([[0, 0], [0, 0]], facecolor='y', alpha=0.3)
    update_polygon(-1)
    ax.add_patch(polygon)

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    def save_fig(event):
        if fname is not None:
            fig.savefig(fname, transparent=True)

    fig.canvas.mpl_connect('axes_leave_event', save_fig)

    plt.show()


if __name__ == '__main__':
    # 'images/general/quad_cells.pdf' 'images/general/Delaunay_and_Voronoi.pdf'
    quad_cells('meshes/msh/rectangle_1_triangle.msh', 'meshes/msh/rectangle_1_quadrangle.msh', 'meshes/msh/rectangle_1_small_quadrangle.msh', None, figsize=np.array((1, 0.75)) * 5, marker_size=4)

    #approximation_cells('meshes/msh/rectangle_1_triangle.msh', 'meshes/msh/rectangle_1_quadrangle.msh', 'meshes/msh/rectangle_1_small_quadrangle.msh', None, figsize=np.array((1, 0.75)) * 5, marker_size=4)

    # fig, ax = plt.subplots(dpi=300, tight_layout=True)
    # d = PolyCollection(Delaunay, closed=True, facecolors='white', edgecolors='blue')
    # v = PolyCollection(Voronoi, closed=False, facecolors='white', edgecolors='red', linestyles='--')

    # ax.add_collection(d)
    # ax.add_collection(v)

    # plt.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo')
    # plt.plot(Voronoi_points[:, 0], Voronoi_points[:, 1], 'ro')

    # #ax.autoscale(tight=True)
    # ax.axis('scaled')
    # #ax.set_aspect('equal', 'box')

    # ax.set_axis_off()

    # plt.show()
    # #plt.savefig(figname, transparent=True)