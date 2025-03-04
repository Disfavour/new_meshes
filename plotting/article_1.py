import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import sys
sys.path.append('mesh_generation')
import utility
import meshio


def Delaunay_and_Voronoi(triangle_mesh):
    Delaunay, Voronoi = [], []
    Delaunay_points, Voronoi_points = [], []

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
    return map(np.array, (Delaunay, Voronoi, Delaunay_points, Voronoi_points))


def get_nodes_and_cells(mesh_file):
    mesh = meshio.read(mesh_file)

    node_coords = mesh.points[:, :2]
    cells = np.array([[node_coords[node] for node in cell] for cell in mesh.cells[0].data])

    return node_coords, cells


def Delaunay_mesh(triangle_mesh, fname, figsize=np.array((1, 0.75)) * 5, marker_size=4, dpi=300):
    Delaunay, Voronoi, Delaunay_points, Voronoi_points = Delaunay_and_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    pc = PolyCollection(Delaunay, closed=True, facecolors='none', edgecolors='blue')

    ax.add_collection(pc)

    ax.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def Voronoi_mesh(triangle_mesh, fname, figsize=np.array((1, 0.75)) * 5, marker_size=4, dpi=300):
    Delaunay, Voronoi, Delaunay_points, Voronoi_points = Delaunay_and_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    pc = PolyCollection(Voronoi, closed=True, facecolors='none', edgecolors='red')

    ax.add_collection(pc)

    ax.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_points[:, 0], Voronoi_points[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


def quadrangle_mesh(quadrangle_mesh, triangle_mesh, fname, figsize=np.array((1, 0.75)) * 5, marker_size=4, dpi=300):
    quad_nodes, quad_cells = get_nodes_and_cells(quadrangle_mesh)
    Delaunay, Voronoi, Delaunay_points, Voronoi_points = Delaunay_and_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    pc = PolyCollection(quad_cells, closed=True, facecolors='none', edgecolors='m')

    ax.add_collection(pc)

    ax.plot(Delaunay_points[:, 0], Delaunay_points[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_points[:, 0], Voronoi_points[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.savefig(fname, transparent=True)


if __name__ == '__main__':
    import os

    article_dir = os.path.join('images', 'article_1')
    os.makedirs(article_dir, exist_ok=True)

    triangle_mesh = os.path.join('meshes', 'msh', 'rectangle_1_triangle.msh')
    
    #Delaunay_mesh(triangle_mesh, os.path.join(article_dir, 'Delaunay.pdf'))
    #Voronoi_mesh(triangle_mesh, os.path.join(article_dir, 'Voronoi.pdf'))
    quadrangle_mesh(os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'), triangle_mesh, os.path.join(article_dir, 'Quadrangle.pdf'))
