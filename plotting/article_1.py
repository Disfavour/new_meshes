import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import sys
sys.path.append('mesh_generation')
sys.path.append('computations')
import utility
import meshio
import matplotlib.patches as patches
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
import poisson_article_Dirichlet, poisson_article_mixed
#import matplotlib
#matplotlib.rcParams.update({'font.size': 20})


def get_Voronoi(triangle_mesh):
    Voronoi_cells, Voronoi_nodes = [], []

    gmsh.initialize()

    gmsh.open(triangle_mesh)
    
    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    gmsh.model.mesh.createEdges()

    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)

    circumcenters = [utility.circumcenter(*(node_coords[node - one] for node in nodes)) for nodes in triangle_nodes.reshape((-1, 3))]
    triangle_to_circumcenter = {triangle: circumcenter for triangle, circumcenter in zip(triangle_tags, circumcenters)}
    Voronoi_nodes = circumcenters.copy()

    edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type)
    edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
    triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
    edge_to_triangles = utility.reverse_dict(triangle_to_edges)

    edge_tags, edge_node_tags = gmsh.model.mesh.get_all_edges()
    for edge, edge_nodes in zip(edge_tags, edge_node_tags.reshape(-1, 2)):
        adjacent_triangles = edge_to_triangles[edge]

        if len(adjacent_triangles) == 2:
            Voronoi_cells.append([triangle_to_circumcenter[triangle] for triangle in adjacent_triangles])
        else:
            Voronoi_nodes.append(np.array([node_coords[node - one] for node in edge_nodes]).sum(axis=0) / 2)
            Voronoi_cells.append([triangle_to_circumcenter[adjacent_triangles[0]], Voronoi_nodes[-1]])

    gmsh.finalize()
    return np.array(Voronoi_cells)[:, :, :2], np.array(Voronoi_nodes)[:, :2]


def get_cells_and_nodes(mesh_file):
    mesh = meshio.read(mesh_file)

    nodes = mesh.points[:, :2]
    cells = np.array([[nodes[node] for node in cell] for cell in mesh.cells[0].data])

    min_x, min_y = nodes.min(axis=0)
    max_x, max_y = nodes.max(axis=0)

    boundary = np.array([
        [
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y],
        ]
    ])

    return cells, nodes, boundary


def get_centroids(cells):
    return cells.sum(axis=1) / 3


def Delaunay_mesh(triangle_mesh, fname, figsize, marker_size):
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(Delaunay_cells, closed=True, facecolors='none', edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo')#, ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def Voronoi_mesh(triangle_mesh, fname, figsize, marker_size):
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(Voronoi_cells, closed=True, facecolors='none', edgecolors='red')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def quadrangle_mesh(quadrangle_mesh, triangle_mesh, fname, figsize, marker_size):
    quad_cells, quad_nodes, boundary = get_cells_and_nodes(quadrangle_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(quad_cells, closed=True, facecolors='none', edgecolors='m')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')
    
    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def Delaunay_and_Voronoi_mesh(triangle_mesh, fname, figsize, marker_size):
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    Delaunay_pc = PolyCollection(Delaunay_cells, closed=True, facecolors='none', edgecolors='blue')
    Voronoi_pc = PolyCollection(Voronoi_cells, closed=True, facecolors='none', edgecolors='red')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(Voronoi_pc)
    ax.add_collection(Delaunay_pc)
    ax.add_collection(boundary_pc)

    ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def one_triangle(triangle_mesh, triangle_tag, fname, figsize, marker_size, radius_multiplier):
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    facecolors = ['none' if i != triangle_tag else (0, 0, 1, 0.2) for i in range(Delaunay_cells.shape[0])]
    pc = PolyCollection(Delaunay_cells, closed=True, facecolors=facecolors, edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def triangles_3(circumcenter_mesh, triangle_mesh, triangle_tag, small_triangle_tag, fname, figsize, marker_size, radius_multiplier, ntriangles):
    circumcenter_cells, circumcenter_nodes, boundary = get_cells_and_nodes(circumcenter_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)

    #facecolors = ['none' if i != small_triangle_tag else (0, 0, 1, 0.2) for i in range(circumcenter_cells.shape[0])]

    #facecolors = ['none' if i not in range(triangle_tag*3, (triangle_tag + 1)*3) else (0, 0, 1, 0.2) for i in range(circumcenter_cells.shape[0])]
    #facecolors = ['none' if i not in range(triangle_tag*6, (triangle_tag + 1)*6) else (0, 0, 1, 0.2) for i in range(circumcenter_cells.shape[0])] 

    # facecolors = ['none' if i not in range(triangle_tag*3, (triangle_tag + 1)*3) else (0, 0, 1, 0.2) for i in range(circumcenter_cells.shape[0])]
    # facecolors[small_triangle_tag] = (0, 0, 1, 0.4)

    #triangle6 v3
    facecolors = ['none' if i not in range(triangle_tag*ntriangles, (triangle_tag + 1)*ntriangles) else (0, 0, 1, 0.2) for i in range(circumcenter_cells.shape[0])]
    facecolors[small_triangle_tag] = (0, 0, 1, 0.4)

    pc = PolyCollection(circumcenter_cells, closed=True, facecolors=facecolors, edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    #Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Delaunay_lines = ax.plot(circumcenter_nodes[:, 0], circumcenter_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def one_triangle_bubble(triangle_mesh, triangle_tag, fname, figsize, marker_size, radius_multiplier):
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = get_centroids(Delaunay_cells)

    fig, ax = plt.subplots(figsize=figsize)
    facecolors = ['none' if i != triangle_tag else (0, 0, 1, 0.2) for i in range(Delaunay_cells.shape[0])]
    pc = PolyCollection(Delaunay_cells, closed=True, facecolors=facecolors, edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    #Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'bo', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    #[o.set_clip_path(circle) for o in Voronoi_lines]
    [o.set_clip_path(circle) for o in centroids_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def one_triangle_order_2(uniform_split_mesh, triangle_mesh, triangle_tag, fname, figsize, marker_size, radius_multiplier):
    uniform_split_mesh_cells, uniform_split_mesh_nodes, boundary = get_cells_and_nodes(uniform_split_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)
    facecolors = ['none' if i != triangle_tag else (0, 0, 1, 0.2) for i in range(Delaunay_cells.shape[0])]
    pc = PolyCollection(Delaunay_cells, closed=True, facecolors=facecolors, edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    triangle6_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in triangle6_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def one_triangle_order_2_bubble(uniform_split_mesh, triangle_mesh, triangle_tag, fname, figsize, marker_size, radius_multiplier):
    uniform_split_mesh_cells, uniform_split_mesh_nodes, boundary = get_cells_and_nodes(uniform_split_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = get_centroids(Delaunay_cells)

    #edge_centers = 

    fig, ax = plt.subplots(figsize=figsize)
    facecolors = ['none' if i != triangle_tag else (0, 0, 1, 0.2) for i in range(Delaunay_cells.shape[0])]
    pc = PolyCollection(Delaunay_cells, closed=True, facecolors=facecolors, edgecolors='blue')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    #Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Delaunay_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo', ms=marker_size)
    #Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'bo', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    #[o.set_clip_path(circle) for o in Voronoi_lines]
    [o.set_clip_path(circle) for o in centroids_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


# решил взять просто радиус описанного треугольника
# def image_10_1(triangle_mesh, triangle_tag, fname, figsize, marker_size, center, radius):
#     Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
#     Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

#     fig, ax = plt.subplots(figsize=figsize)

#     facecolors = ['none' if i != triangle_tag else (0, 0, 1, 0.2) for i in range(Delaunay_cells.shape[0])]
#     pc = PolyCollection(Delaunay_cells, closed=True, facecolors=facecolors, edgecolors='blue')
#     boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

#     ax.add_collection(pc)
#     ax.add_collection(boundary_pc)

#     Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
#     Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

#     #circumcenter = Voronoi_nodes[triangle_tag]
#     #radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

#     circle = patches.Circle(center, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
#     #ax.add_patch(circle)

#     pc.set_clip_path(circle)
#     boundary_pc.set_clip_path(circle)
#     [o.set_clip_path(circle) for o in Delaunay_lines]
#     [o.set_clip_path(circle) for o in Voronoi_lines]

#     min_x, min_y = center - radius
#     max_x, max_y = center + radius

#     ax.axis([min_x, max_x, min_y, max_y])
#     ax.set_aspect(1)

#     #ax.axis('scaled')
#     ax.set_axis_off()

#     fig.tight_layout()

#     fig.savefig(fname, transparent=True)
#     plt.close()


def image_10_2(quad_mesh, triangle_mesh, triangle_tag, quadrangle_tag, fname, figsize, marker_size, radius_multiplier):
    quad_cells, quad_nodes, boundary = get_cells_and_nodes(quad_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)

    facecolors = ['none' if i != quadrangle_tag else (0, 0, 1, 0.2) for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def image_10_3(split_quad_mesh, triangle_mesh, triangle_tag, triangle_tag2, fname, figsize, marker_size, radius_multiplier):
    split_quad_cells, split_quad_nodes, boundary = get_cells_and_nodes(split_quad_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)

    #facecolors = ['none' if i != triangle_tag // 2 else (0, 0, 1, 0.2) for i in range(quad_cells.shape[0])]
    #pc_quad = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='none')
    facecolors = [(0, 0, 1, 0.2) if i in (triangle_tag // 2 * 2, triangle_tag // 2 * 2 + 1) else 'none' for i in range(split_quad_cells.shape[0])]
    facecolors[triangle_tag] = (0, 0, 1, 0.4)
    pc_split = PolyCollection(split_quad_cells, closed=True, facecolors=facecolors, edgecolors='m')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    #ax.add_collection(pc_quad)
    ax.add_collection(pc_split)
    ax.add_collection(boundary_pc)

    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag2]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag2][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    #pc_quad.set_clip_path(circle)
    pc_split.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def image_13_2(quad_mesh, small_quad_mesh, triangle_mesh, quad_tag, triangle_tag, fname, figsize, marker_size, radius_multiplier):
    quad_cells, quad_nodes, boundary = get_cells_and_nodes(quad_mesh)
    small_quad_cells, small_quad_nodes, boundary = get_cells_and_nodes(small_quad_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize)

    facecolors = ['none' if i != quad_tag else (0, 0, 1, 0.2) for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    bubble_lines = ax.plot(small_quad_nodes[:, 0], small_quad_nodes[:, 1], 'bo', ms=marker_size)
    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in bubble_lines]
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def image_13_3(small_quad_mesh, triangle_mesh, quadrangle_tag, fname, figsize, marker_size, radius_multiplier):
    quad_cells, quad_nodes, boundary = get_cells_and_nodes(small_quad_mesh)
    Delaunay_cells, Delaunay_nodes, boundary = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    triangle_tag = quadrangle_tag // 3

    fig, ax = plt.subplots(figsize=figsize)

    #facecolors = ['none' if i != triangle_tag // 2 else (0, 0, 1, 0.2) for i in range(quad_cells.shape[0])]
    #pc_quad = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='none')
    facecolors = [(0, 0, 1, 0.2) if i in range(triangle_tag * 3, (triangle_tag + 1) * 3) else 'none' for i in range(quad_cells.shape[0])]
    facecolors[quadrangle_tag] = (0, 0, 1, 0.4)
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='b')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k')

    #ax.add_collection(pc_quad)
    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    quad_lines = ax.plot(quad_nodes[:, 0], quad_nodes[:, 1], 'bo', ms=marker_size)
    Delaunay_lines = ax.plot(Delaunay_nodes[:, 0], Delaunay_nodes[:, 1], 'bo', ms=marker_size)
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro', ms=marker_size)
    
    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - Delaunay_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)
    #ax.add_patch(circle)

    #pc_quad.set_clip_path(circle)
    pc.set_clip_path(circle)
    boundary_pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in quad_lines]
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)

    #ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(fname, transparent=True)
    plt.close()


def image_4(k, fnames, figsize, scale_limits):
    k = ufl.as_matrix(k)
    
    lagrange = basix.ufl.element("Lagrange", 'triangle', 1)
    enriched_bubble = basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'triangle', 1), basix.ufl.element("Bubble", 'triangle', 3)])

    data = []
    for i in range(1, 11):
        data.append([])
        
        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, lagrange, k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_circumcenter.msh')
        results = poisson_article_mixed.solve(mesh_name, lagrange, k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, enriched_bubble, k)
        data[-1].append(results)
    
    data = np.array(data)

    for fname, error, i in zip(fnames, ('$L_2$', r'$L_{\infty}$', '$H_0^1$'), range(3, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for j, lw, ms in zip(range(data.shape[1]), [1.5, 1.5*3, 1.5], [6, 6*np.sqrt(3), 6]):
            ax.plot(data[:, j, 0], data[:, j, i], '-o', lw=lw, ms=ms)
        
        ax.set_xlabel("$M$")
        ax.set_ylabel(error)
        ax.grid()

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(['а', 'б', 'в'])

        ymin, ymax = data[:, :, i].min(), data[:, :, i].max()
        ymin_degree, ymax_degree = np.floor(np.log10(ymin)), np.ceil(np.log10(ymax))
        ymin = 10 ** ymin_degree
        ymax = 10 ** ymax_degree
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([10 ** i for i in range(int(ymin_degree), int(ymax_degree) + 1)])

        #xmin, xmax, ymin, ymax = ax.axis()
        #ax.axis([xmin, xmax, ymin / scale_limits, ymax * scale_limits])

        fig.tight_layout()
        fig.savefig(fname, transparent=True)
        plt.close()


def image_5(k, fnames, figsize, scale_limits):
    k = ufl.as_matrix(k)

    data = []
    for i in range(1, 11):
        data.append([])
        
        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 2), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_circumcenter_6.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'triangle', 2), basix.ufl.element("Bubble", 'triangle', 3)]), k)
        data[-1].append(results)
    
    data = np.array(data)

    for fname, error, i in zip(fnames, ('$L_2$', r'$L_{\infty}$', '$H_0^1$'), range(3, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for j, lw, ms in zip(range(data.shape[1]), [1.5, 1.5, 1.5], [6, 6, 6]):
            ax.plot(data[:, j, 0], data[:, j, i], '-o', lw=lw, ms=ms)
        
        ax.set_xlabel("$M$")
        ax.set_ylabel(error)
        ax.grid()

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(['а', 'б', 'в'])

        ymin, ymax = data[:, :, i].min(), data[:, :, i].max()
        ymin_degree, ymax_degree = np.floor(np.log10(ymin)), np.ceil(np.log10(ymax))
        ymin = 10 ** ymin_degree
        ymax = 10 ** ymax_degree
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([10 ** i for i in range(int(ymin_degree), int(ymax_degree) + 1)])

        #xmin, xmax, ymin, ymax = ax.axis()
        #ax.axis([xmin, xmax, ymin / scale_limits, ymax * scale_limits])

        fig.tight_layout()
        fig.savefig(fname, transparent=True)
        plt.close()


def image_6(k, fnames, figsize, scale_limits):
    k = ufl.as_matrix(k)

    data = []
    for i in range(1, 11):
        data.append([])
        
        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_split_quadrangles.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)
    
    data = np.array(data)

    for fname, error, i in zip(fnames, ('$L_2$', r'$L_{\infty}$', '$H_0^1$'), range(3, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for j, lw, ms in zip(range(data.shape[1]), [1.5, 1.5, 1.5], [6, 6, 6]):
            ax.plot(data[:, j, 0], data[:, j, i], '-o', lw=lw, ms=ms)
        
        ax.set_xlabel("$M$")#, fontsize=20)
        ax.set_ylabel(error)#, fontsize=20)
        ax.grid()

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(['а', 'б', 'в'])#, fontsize=20)

        ymin, ymax = data[:, :, i].min(), data[:, :, i].max()
        ymin_degree, ymax_degree = np.floor(np.log10(ymin)), np.ceil(np.log10(ymax))
        ymin = 10 ** ymin_degree
        ymax = 10 ** ymax_degree
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([10 ** i for i in range(int(ymin_degree), int(ymax_degree) + 1)])

        # if miny / ymin < scale_limits:
        #     ymin = 10 ** (np.floor(np.log10(miny)) - 1)
        # if ymax / maxy < scale_limits:
        #     ymax = 10 ** (np.ceil(np.log10(maxy)) + 1)

        

        #xmin, xmax, ymin, ymax = ax.axis()
        #ax.axis([xmin, xmax, ymin / scale_limits, ymax * scale_limits])

        fig.tight_layout()
        fig.savefig(fname, transparent=True)#, bbox_inches="tight")
        plt.close()


def image_13(k, fnames, figsize, scale_limits):
    k = ufl.as_matrix(k)

    data = []
    for i in range(1, 11):
        data.append([])
        
        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'quadrilateral', 1), basix.ufl.element("Bubble", 'quadrilateral', 3)]), k)
        data[-1].append(results)

        mesh_name = os.path.join('meshes', 'msh', f'rectangle_{i}_small_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)
    
    data = np.array(data)

    for fname, error, i in zip(fnames, ('$L_2$', r'$L_{\infty}$', '$H_0^1$'), range(3, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for j, lw, ms in zip(range(data.shape[1]), [1.5, 1.5, 1.5], [6, 6, 6]):
            ax.plot(data[:, j, 0], data[:, j, i], '-o', lw=lw, ms=ms)
        
        ax.set_xlabel("$M$")#, fontsize=20)
        ax.set_ylabel(error)#, fontsize=20)
        ax.grid()

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(['а', 'б', 'в'])#, fontsize=20)

        ymin, ymax = data[:, :, i].min(), data[:, :, i].max()
        ymin_degree, ymax_degree = np.floor(np.log10(ymin)), np.ceil(np.log10(ymax))
        ymin = 10 ** ymin_degree
        ymax = 10 ** ymax_degree
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([10 ** i for i in range(int(ymin_degree), int(ymax_degree) + 1)])

        # if miny / ymin < scale_limits:
        #     ymin = 10 ** (np.floor(np.log10(miny)) - 1)
        # if ymax / maxy < scale_limits:
        #     ymax = 10 ** (np.ceil(np.log10(maxy)) + 1)

        

        #xmin, xmax, ymin, ymax = ax.axis()
        #ax.axis([xmin, xmax, ymin / scale_limits, ymax * scale_limits])

        fig.tight_layout()
        fig.savefig(fname, transparent=True)#, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    import os

    article_dir = os.path.join('images', 'article_1')
    os.makedirs(article_dir, exist_ok=True)

    triangle_mesh = os.path.join('meshes', 'msh', 'rectangle_1_triangle.msh')
    triangle6_mesh = os.path.join('meshes', 'msh', 'rectangle_1_circumcenter_6.msh')
    uniform_split_mesh = os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh')

    figsize = np.array((1, 0.75)) * 5
    marker_size = 6

    figsize_circle = np.array((1, 1)) * 5
    radius_multiplier = 1.2

    triangle_tag = 12
    small_triangle_tag = triangle_tag*3 + 1
    very_small_triangle_tag = 12*6 + 2
    
    Delaunay_mesh(triangle_mesh, os.path.join(article_dir, 'Delaunay.pdf'), figsize, marker_size)
    Voronoi_mesh(triangle_mesh, os.path.join(article_dir, 'Voronoi.pdf'), figsize, marker_size)
    quadrangle_mesh(os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'), triangle_mesh, os.path.join(article_dir, 'Quadrangle.pdf'), figsize, marker_size)
    Delaunay_and_Voronoi_mesh(triangle_mesh, os.path.join(article_dir, 'Delaunay_and_Voronoi.pdf'), figsize, marker_size)

    one_triangle(triangle_mesh, triangle_tag, os.path.join(article_dir, 'one_triangle.pdf'), figsize_circle, marker_size, radius_multiplier)
    triangles_3(os.path.join('meshes', 'msh', 'rectangle_1_circumcenter.msh'), triangle_mesh, triangle_tag, small_triangle_tag,
                           os.path.join(article_dir, 'triangles_3.pdf'), figsize_circle, marker_size, radius_multiplier, 3)
    one_triangle_bubble(triangle_mesh, triangle_tag, os.path.join(article_dir, 'one_triangle_bubble.pdf'), figsize_circle, marker_size, radius_multiplier)

    one_triangle_order_2(uniform_split_mesh, triangle_mesh, triangle_tag, os.path.join(article_dir, 'one_triangle_order_2.pdf'), figsize_circle, marker_size, radius_multiplier)
    triangles_3(triangle6_mesh, triangle_mesh, triangle_tag, very_small_triangle_tag,
                           os.path.join(article_dir, 'triangles_6.pdf'), figsize_circle, marker_size, radius_multiplier, 6)
    one_triangle_order_2_bubble(uniform_split_mesh, triangle_mesh, triangle_tag, os.path.join(article_dir, 'one_triangle_order_2_bubble.pdf'), figsize_circle, marker_size, radius_multiplier)

    # center, radius = np.array([0.66888747, 0.28479431]), 0.21235271386358628 * radius_multiplier
    #center, radius = np.array([0.669, 0.285]), 0.212 * radius_multiplier

    # image_10_1(triangle_mesh, triangle_tag, os.path.join(article_dir, '10-1.pdf'), figsize_circle, marker_size, center, radius)
    # image_10_2(os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'), triangle_mesh, 18, os.path.join(article_dir, '10-2.pdf'), figsize_circle, marker_size, center, radius)
    # image_10_3(os.path.join('meshes', 'msh', 'rectangle_1_split_quadrangles.msh'),
    #            os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'),
    #            triangle_mesh, 36, os.path.join(article_dir, '10-3.pdf'), figsize_circle, marker_size, center, radius)
    
    #image_10_1(triangle_mesh, triangle_tag, os.path.join(article_dir, '10-1.pdf'), figsize_circle, marker_size, radius_multiplier)
    image_10_2(os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'), triangle_mesh, triangle_tag, 31, os.path.join(article_dir, '10-2.pdf'), figsize_circle, marker_size, radius_multiplier)
    image_10_3(os.path.join('meshes', 'msh', 'rectangle_1_split_quadrangles.msh'),
               triangle_mesh, 63, triangle_tag, os.path.join(article_dir, '10-3.pdf'), figsize_circle, marker_size, radius_multiplier)
    
    image_13_2(os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh'), os.path.join('meshes', 'msh', 'rectangle_1_small_quadrangle.msh'),
               triangle_mesh, 31, triangle_tag, os.path.join(article_dir, '13-2.pdf'), figsize_circle, marker_size, radius_multiplier)
    image_13_3(os.path.join('meshes', 'msh', 'rectangle_1_small_quadrangle.msh'), triangle_mesh, 37, os.path.join(article_dir, '13-3.pdf'), figsize_circle, marker_size, radius_multiplier)

    figsize=np.array((6.4, 3.6))# / 1.6
    font_size = 10 * 2
    import matplotlib
    matplotlib.rcParams['font.size'] = '16'

    scale_limits = 1.5 # не используется

    ks = [
        [
            [1, 0],
            [0, 1]
        ],
        [
            [1, 30],
            [30, 1000]
        ]
    ]
    for i, k in enumerate(ks, 1):
        j = 4
        fnames = [
            os.path.join(article_dir, f'image_{j}_L2_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_Lmax_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_H01_k{i}.pdf')
        ]
        image_4(k, fnames, figsize, scale_limits)

        j = 5
        fnames = [
            os.path.join(article_dir, f'image_{j}_L2_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_Lmax_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_H01_k{i}.pdf')
        ]
        image_5(k, fnames, figsize, scale_limits)

        j = 6
        fnames = [
            os.path.join(article_dir, f'image_{j}_L2_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_Lmax_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_H01_k{i}.pdf')
        ]
        image_6(k, fnames, figsize, scale_limits)

        j = 13
        fnames = [
            os.path.join(article_dir, f'image_{j}_L2_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_Lmax_k{i}.pdf'),
            os.path.join(article_dir, f'image_{j}_H01_k{i}.pdf')
        ]
        image_13(k, fnames, figsize, scale_limits)
    
    

    #get_Voronoi(triangle_mesh)
    # точки вороного соответствуют номерам треугольников (по построению)
    # мб просто отделить краевые узлы вороного, подумать какие операции вообще будут нужны

    # построить фул сетки, а потом вырезать кружочки?
