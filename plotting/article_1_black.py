import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import numpy as np
import meshio
import os
import gmsh
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
import sys
sys.path.append('mesh_generation')
sys.path.append('computations')
import utility
import poisson_article_Dirichlet, poisson_article_mixed


# default figsize [6.4, 4.8] is ok
figsize = np.array([6.4, 4.8]) / 3
figsize_circle = np.array((1, 1)) * 1.6
#figsize_plot = np.array([6.4, 4.8]) / 1.6
figsize_plot = np.array((6.4, 3.6)) / 1.2 
dpi = 600
(min_x, min_y), (max_x, max_y) = (0, 0), (1, 0.75)
boundary = np.array([
    [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
])
triangle_mesh = os.path.join('meshes', 'msh', 'rectangle_1_triangle.msh')
quadrangle_mesh = os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh')

article_dir = os.path.join('images', 'article_1_black')
os.makedirs(article_dir, exist_ok=True)

triangle_tag = 12
radius_multiplier = 1.2


def get_cells_and_nodes(mesh_file):
    mesh = meshio.read(mesh_file)

    nodes = mesh.points[:, :2]
    cells = np.array([[nodes[node] for node in cell] for cell in mesh.cells[0].data])

    return cells, nodes


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


def image_1():
    cells, nodes = get_cells_and_nodes(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pc = PolyCollection(cells, closed=True, facecolors='none', edgecolors='k')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k', linewidths=2)

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(nodes[:, 0], nodes[:, 1], 'ko')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '1.png'), transparent=True)
    plt.close()


def image_2():
    cells, nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pc = PolyCollection(Voronoi_cells, closed=False, facecolors='none', edgecolors='k')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k', linewidths=2)

    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(nodes[:, 0], nodes[:, 1], 'ko')
    ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')
    
    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '2.png'), transparent=True)
    plt.close()


def image_3():
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pc = PolyCollection(quad_cells, closed=True, facecolors='none', edgecolors='k')
    boundary_pc = PolyCollection(boundary, closed=True, facecolors='none', edgecolors='k', linewidths=2)
    
    ax.add_collection(pc)
    ax.add_collection(boundary_pc)

    ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    ax.axis('scaled')
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '3.png'), transparent=True)
    plt.close()


def image_4_1(fname=os.path.join(article_dir, '4-1.png')):
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)
    facecolors = [(0, 0, 0, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    
    ax.axis([min_x, max_x, min_y, max_y])

    ax.set_aspect(1)
    ax.set_axis_off()
    
    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


def image_4_2():
    circumcenter_cells, circumcenter_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_circumcenter.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i in range(triangle_tag*3, (triangle_tag + 1)*3) else 'none' for i in range(circumcenter_cells.shape[0])]
    facecolors[triangle_tag*3 + 1] = (0, 0, 0, 0.6)

    pc = PolyCollection(circumcenter_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '4-2.png'), transparent=True)
    plt.close()


def image_4_3():
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = triangle_cells.sum(axis=1) / 3

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    triangle_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'ko')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in triangle_lines]
    [o.set_clip_path(circle) for o in centroids_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '4-3.png'), transparent=True)
    plt.close()


def image_5_1():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    triangle6_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in triangle6_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '5-1.png'), transparent=True)
    plt.close()


def image_5_2():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    circumcenter_cells, circumcenter_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_circumcenter_6.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i in range(triangle_tag*6, (triangle_tag + 1)*6) else 'none' for i in range(circumcenter_cells.shape[0])]
    facecolors[triangle_tag*6 + 2] = (0, 0, 0, 0.6)

    pc = PolyCollection(circumcenter_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '5-2.png'), transparent=True)
    plt.close()


def image_5_3():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = triangle_cells.sum(axis=1) / 3

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    triangle6_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'ko')
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'ko')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in triangle6_lines]
    [o.set_clip_path(circle) for o in centroids_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '5-3.png'), transparent=True)
    plt.close()


def image_10_1():
    image_4_1(fname=os.path.join(article_dir, '10-1.png'))


def image_10_2(fname=os.path.join(article_dir, '10-2.png')):
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i == 31 else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(fname, transparent=True)
    plt.close()


def image_10_3():
    quad_cells, quad_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_split_quadrangles.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i in range(31 * 2, (31 + 1) * 2) else 'none' for i in range(quad_cells.shape[0])]
    facecolors[63] = (0, 0, 0, 0.6)
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '10-3.png'), transparent=True)
    plt.close()


def image_13_1():
    image_10_2(fname=os.path.join(article_dir, '13-1.png'))


def image_13_2():
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = quad_cells.sum(axis=1) / 4

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i == 31 else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)

    centroid_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'ko')
    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for o in centroid_lines]
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '13-2.png'), transparent=True)
    plt.close()


def image_13_3():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    small_quad_cells, small_quad_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_small_quadrangle.msh'))
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=figsize_circle, dpi=dpi)

    facecolors = [(0, 0, 0, 0.3) if i in (18, 31) else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='k')

    # Результирующая прозрачность=1−(1−α_1)×(1−α_2)
    a = 0.6
    a1 = 0.3
    a2 = 1 - (1 - a)/(1 - a1)

    facecolors = [(0, 0, 0, a2) if i == 37 else 'none' for i in range(small_quad_cells.shape[0])]
    pc2 = PolyCollection(small_quad_cells, closed=True, facecolors=facecolors, edgecolors='k')

    ax.add_collection(pc)
    ax.add_collection(pc2)

    Delaunay_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'ko')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'k^')

    circumcenter = Voronoi_nodes[triangle_tag]
    radius = np.linalg.norm(circumcenter - triangle_cells[triangle_tag][0]) * radius_multiplier

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    pc2.set_clip_path(circle)
    [o.set_clip_path(circle) for o in Delaunay_lines]
    [o.set_clip_path(circle) for o in Voronoi_lines]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius

    ax.axis([min_x, max_x, min_y, max_y])
    ax.set_aspect(1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(article_dir, '13-3.png'), transparent=True)
    plt.close()


def image_67(k, fname, ymin=None, ymax=None):
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

    i = 3

    fig, ax = plt.subplots(figsize=figsize_plot, dpi=dpi)
    
    for j, color, marker in zip(range(data.shape[1]), ((0, 0, 0, 1), (0, 0, 0, 0.5), (0, 0, 0, 1)), ('^', 'D', 'o')):
        if j == 1:
            ax.plot(data[:, j, 0], data[:, j, i], marker + '-k', markersize=6*np.sqrt(3), markerfacecolor=color, markeredgewidth=0)
        else:
            ax.plot(data[:, j, 0], data[:, j, i], marker + '-k')
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(['а', 'б', 'в'], scatteryoffsets=[1, 0.5, 0])

    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)


    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)
    plt.close()


def image_89(k, fname):
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

    i = 3
    fig, ax = plt.subplots(figsize=figsize_plot, dpi=dpi)

    for j, marker in zip(range(data.shape[1]), ('^', 'D', 'o')):
        ax.plot(data[:, j, 0], data[:, j, i], marker+'-k')
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(['а', 'б', 'в'])

    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)
    plt.close()


def image_11_12(k, fname):
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

    i = 3
    fig, ax = plt.subplots(figsize=figsize_plot, dpi=dpi)

    for j, marker in zip(range(data.shape[1]), ('^', 'D', 'o')):
        ax.plot(data[:, j, 0], data[:, j, i], marker+'-k')
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(['а', 'б', 'в'])

    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)
    plt.close()


def image_14_15(k, fname):
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

    i = 3
    fig, ax = plt.subplots(figsize=figsize_plot, dpi=dpi)

    for j, marker in zip(range(data.shape[1]), ('^', 'D', 'o')):
        ax.plot(data[:, j, 0], data[:, j, i], marker+'-k')
    
    ax.set_xlabel("$M$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(['а', 'б', 'в'])

    fig.tight_layout(pad=0.2)
    fig.savefig(fname, transparent=True)
    plt.close()


if __name__ == '__main__':
    # image_1()
    # image_2()
    # image_3()
    # image_4_1()
    # image_4_2()
    # image_4_3()
    # image_5_1()
    # image_5_2()
    # image_5_3()
    # image_10_1()
    # image_10_2()
    # image_10_3()
    # image_13_1()
    # image_13_2()
    # image_13_3()

    ks = [[[1, 0],
           [0, 1]],
           [[1, 30],
            [30, 1000]]]
    
    # ymax=1e-2
    image_67(ks[0], os.path.join(article_dir, '6.png'))
    image_67(ks[1], os.path.join(article_dir, '7.png'))#, ymin=1e-4)

    image_89(ks[0], os.path.join(article_dir, '8.png'))
    image_89(ks[1], os.path.join(article_dir, '9.png'))

    image_11_12(ks[0], os.path.join(article_dir, '11.png'))
    image_11_12(ks[1], os.path.join(article_dir, '12.png'))

    image_14_15(ks[0], os.path.join(article_dir, '14.png'))
    image_14_15(ks[1], os.path.join(article_dir, '15.png'))
