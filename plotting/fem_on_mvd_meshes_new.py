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
sys.path.append('computations')
import poisson_article_mixed
import utility


def read_mesh(quadrangle_mesh):
    gmsh.initialize()
    gmsh.open(quadrangle_mesh)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)

    quadrangle_tags, quad_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    quad_nodes = quad_nodes.reshape(-1, 4) - 1

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    loaded = np.load(quadrangle_mesh.replace('.msh', '.npz'), allow_pickle=True)
    node_groups = loaded['node_groups'].astype(int)
    cell_nodes = loaded['cells'] - 1
    cell_nodes = np.array([nodes.astype(int) for nodes in cell_nodes], dtype=object)

    return node_coords, quad_nodes, node_groups, cell_nodes


def triangle_with_V_nodes(fname, cells_V_coords, nodes_D_coords, nodes_V_coords, node_V, circumcenter, radius):
    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    fc = ['none' if i != node_V else (0, 0, 1, 0.3) for i in range(cells_V_coords.shape[0])]
    cells_V = PolyCollection(cells_V_coords, closed=True, facecolors='none', edgecolors='b')

    cell_V = PolyCollection(cells_V_coords, closed=True, facecolors=(0, 0, 1, 0.3), edgecolors='none')

    ax.add_collection(pc)

    lines = []
    lines.append(ax.plot(nodes_D_coords[:, 0], nodes_D_coords[:, 1], 'bo'))
    lines.append(ax.plot(nodes_V_coords[:, 0], nodes_V_coords[:, 1], 'ro'))

    circle = patches.Circle(circumcenter, radius=radius, color=(1, 0, 0, 0.1), transform=ax.transData)

    pc.set_clip_path(circle)
    [o.set_clip_path(circle) for line in lines for o in line]

    min_x, min_y = circumcenter - radius
    max_x, max_y = circumcenter + radius
    
    ax.axis([min_x, max_x, min_y, max_y])

    ax.set_aspect(1)
    ax.set_axis_off()
    
    fig.savefig(fname, transparent=True)
    plt.close()


def image_4_1(fname):
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)
    facecolors = [(0, 0, 1, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    
    fig.savefig(fname, transparent=True)
    plt.close()


def image_4_2():
    circumcenter_cells, circumcenter_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_circumcenter.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i in range(triangle_tag*3, (triangle_tag + 1)*3) else 'none' for i in range(circumcenter_cells.shape[0])]
    facecolors[triangle_tag*3 + 1] = (0, 0, 1, 0.6)

    pc = PolyCollection(circumcenter_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '4-2.pdf'), transparent=True)
    plt.close()


def image_4_3():
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = triangle_cells.sum(axis=1) / 3

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    triangle_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'bo')

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
    fig.savefig(os.path.join(article_dir, '4-3.pdf'), transparent=True)
    plt.close()


def image_5_1():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    triangle6_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '5-1.pdf'), transparent=True)
    plt.close()


def image_5_2():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    circumcenter_cells, circumcenter_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_circumcenter_6.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i in range(triangle_tag*6, (triangle_tag + 1)*6) else 'none' for i in range(circumcenter_cells.shape[0])]
    facecolors[triangle_tag*6 + 2] = (0, 0, 1, 0.6)

    pc = PolyCollection(circumcenter_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '5-2.pdf'), transparent=True)
    plt.close()


def image_5_3():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = triangle_cells.sum(axis=1) / 3

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i == triangle_tag else 'none' for i in range(triangle_cells.shape[0])]
    pc = PolyCollection(triangle_cells, closed=True, facecolors=facecolors, edgecolors='b')

    ax.add_collection(pc)

    triangle6_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo')
    centroids_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'bo')

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
    fig.savefig(os.path.join(article_dir, '5-3.pdf'), transparent=True)
    plt.close()


def image_10_1():
    image_4_1(fname=os.path.join(article_dir, '10-1.pdf'))


def image_10_2(fname):
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i == 31 else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(fname, transparent=True)
    plt.close()


def image_10_3():
    quad_cells, quad_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_split_quadrangles.msh'))
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i in range(31 * 2, (31 + 1) * 2) else 'none' for i in range(quad_cells.shape[0])]
    facecolors[63] = (0, 0, 1, 0.6)
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')

    ax.add_collection(pc)

    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '10-3.pdf'), transparent=True)
    plt.close()


def image_13_1():
    image_10_2(fname=os.path.join(article_dir, '13-1.pdf'))


def image_13_2():
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)
    centroids = quad_cells.sum(axis=1) / 4

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i == 31 else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')

    ax.add_collection(pc)

    centroid_lines = ax.plot(centroids[:, 0], centroids[:, 1], 'bo')
    Delaunay_lines = ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '13-2.pdf'), transparent=True)
    plt.close()


def image_13_3():
    uniform_split_mesh_cells, uniform_split_mesh_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_uniform_split.msh'))
    small_quad_cells, small_quad_nodes = get_cells_and_nodes(os.path.join('meshes', 'msh', 'rectangle_1_small_quadrangle.msh'))
    quad_cells, quad_nodes = get_cells_and_nodes(quadrangle_mesh)
    triangle_cells, triangle_nodes = get_cells_and_nodes(triangle_mesh)
    Voronoi_cells, Voronoi_nodes = get_Voronoi(triangle_mesh)

    fig, ax = plt.subplots(figsize=utility.get_figsize(1, 1), constrained_layout=True)

    facecolors = [(0, 0, 1, 0.3) if i in (18, 31) else 'none' for i in range(quad_cells.shape[0])]
    pc = PolyCollection(quad_cells, closed=True, facecolors=facecolors, edgecolors='m')

    # Результирующая прозрачность=1−(1−α_1)×(1−α_2)
    a = 0.6
    a1 = 0.3
    a2 = 1 - (1 - a)/(1 - a1)

    facecolors = [(0, 0, 1, a2) if i == 37 else 'none' for i in range(small_quad_cells.shape[0])]
    pc2 = PolyCollection(small_quad_cells, closed=True, facecolors=facecolors, edgecolors='m')

    ax.add_collection(pc)
    ax.add_collection(pc2)

    Delaunay_lines = ax.plot(uniform_split_mesh_nodes[:, 0], uniform_split_mesh_nodes[:, 1], 'bo')
    Voronoi_lines = ax.plot(Voronoi_nodes[:, 0], Voronoi_nodes[:, 1], 'ro')

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
    fig.savefig(os.path.join(article_dir, '13-3.pdf'), transparent=True)
    plt.close()


def image_6_7(k, fname1, fname2, legend, ymin=None, ymax=None):
    k = ufl.as_matrix(k)
    
    lagrange = basix.ufl.element("Lagrange", 'triangle', 1)
    enriched_bubble = basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'triangle', 1), basix.ufl.element("Bubble", 'triangle', 3)])

    data = []
    for i in range(n):
        data.append([])
        
        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, lagrange, k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle_circumcenter_3.msh')
        results = poisson_article_mixed.solve(mesh_name, lagrange, k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, enriched_bubble, k)
        data[-1].append(results)
    
    data = np.array(data)


    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    
    for j, ms in zip(range(data.shape[1]), (6*np.sqrt(4), 6*np.sqrt(2), 6)):
        if j == 1:
            ax.plot(data[:, j, 0], data[:, j, 3], 'o-', markersize=6*np.sqrt(2))
        else:
            ax.plot(data[:, j, 0], data[:, j, 3], 'o-')
        
        #ax.plot(data[:, j, 1], data[:, j, i], 'o-')
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.legend(legend, scatteryoffsets=[1, 0.5, 0])
    ax.legend(legend)

    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    fig.savefig(fname1, transparent=True)


    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    
    for j, ms in zip(range(data.shape[1]), (6*np.sqrt(4), 6*np.sqrt(2), 6)):
        ax.plot(data[:, 0, 0], data[:, j, 3], 'o-', markersize=ms)
    
    ax.set_xlabel("$N_D$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.legend(legend, scatteryoffsets=[1, 0.5, 0])
    ax.legend(legend)

    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    fig.savefig(fname2, transparent=True)
    plt.close()


def image_8_9(k, fname1, fname2, legend, ymin=None, ymax=None):
    k = ufl.as_matrix(k)

    data = []
    for i in range(n):
        data.append([])
        
        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 2), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle_circumcenter_6.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'triangle', 2), basix.ufl.element("Bubble", 'triangle', 3)]), k)
        data[-1].append(results)
    
    data = np.array(data)

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    for j in range(data.shape[1]):
        ax.plot(data[:, j, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname1, transparent=True)


    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)

    for j in range(data.shape[1]):
        ax.plot(data[:, 0, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N_D$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname2, transparent=True)

    plt.close()


def image_11_12(k, fname1, fname2, legend, ymin=None, ymax=None):
    k = ufl.as_matrix(k)

    data = []
    for i in range(n):
        data.append([])
        
        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_triangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_quadrangle_split.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'triangle', 1), k)
        data[-1].append(results)
    
    data = np.array(data)

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    for j in range(data.shape[1]):
        ax.plot(data[:, j, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname1, transparent=True)


    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    for j in range(data.shape[1]):
        ax.plot(data[:, 0, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N_D$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname2, transparent=True)
    plt.close()


def image_14_15(k, fname1, fname2, legend, ymin=None, ymax=None):
    k = ufl.as_matrix(k)

    data = []
    for i in range(n):
        data.append([])
        
        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'quadrilateral', 1), basix.ufl.element("Bubble", 'quadrilateral', 2)]), k)
        data[-1].append(results)

        mesh_name = os.path.join(meshes_dir, f'rectangle_{i}_small_quadrangle.msh')
        results = poisson_article_mixed.solve(mesh_name, basix.ufl.element("Lagrange", 'quadrilateral', 1), k)
        data[-1].append(results)
    
    data = np.array(data)

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    for j in range(data.shape[1]):
        ax.plot(data[:, j, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname1, transparent=True)

    fig, ax = plt.subplots(figsize=utility.get_figsize_2_columns_default(), constrained_layout=True)
    for j in range(data.shape[1]):
        ax.plot(data[:, 0, 0], data[:, j, 3], 'o-')
    
    ax.set_xlabel("$N_{MVD}$")
    ax.set_ylabel(r'$\varepsilon$')
    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(legend)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.savefig(fname2, transparent=True)

    plt.close()


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 11


    article_dir = os.path.join('images', 'fem_on_mvd_meshes')
    os.makedirs(article_dir, exist_ok=True)

    triangle_tag = 12
    radius_multiplier = 1.2

    n = 22
    meshes_dir = os.path.join('meshes', 'rectangle')

    quadrangle_mesh = 'meshes/rectangle/rectangle_0_quadrangle.msh'
    coords, quads, groups, cells = read_mesh(quadrangle_mesh)

    nodes_D = np.concatenate((np.arange(groups[0]), np.arange(groups[1], groups[2])))
    nodes_D_coords = coords[nodes_D]

    nodes_V = np.concatenate((np.arange(groups[0], groups[1]), np.arange(groups[2], groups[3])))
    nodes_V_coords = coords[nodes_V]

    cells_D = np.concatenate((cells[:groups[0]], cells[groups[1]:groups[2]]))
    cells_D_coords = [coords[cell] for cell in cells_D]

    cells_V = np.stack((*cells[groups[0]:groups[1]], *cells[groups[2]:]))
    cells_V_coords = coords[cells_V]

    node_V = 22 # triangle_tag
    circumcenter = coords[node_V]
    triangle_node_coords = coords[cells[node_V]]
    radius = np.linalg.norm(circumcenter - triangle_node_coords[0]) * 1.2


    triangle_with_V_nodes('3-1.pdf', cells_V_coords, nodes_D_coords, nodes_V_coords, node_V, circumcenter, radius)


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
    # image_6_7(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '6-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '7-1.pdf'), legend, 1e-6, 2e-2)
    # image_6_7(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '6-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '7-2.pdf'), legend, 1e-6, 2e-2)

    # image_8_9(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '8-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '9-1.pdf'), legend, 1e-10, 1e-2)
    # image_8_9(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '8-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '9-2.pdf'), legend, 1e-10, 1e-2)

    # image_11_12(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '11-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '12-1.pdf'), legend, 1e-7, 2e-2)
    # image_11_12(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '11-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '12-2.pdf'), legend, 1e-7, 2e-2)

    # image_14_15(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '14-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '15-1.pdf'), legend, 1e-7, 1e-2)
    # image_14_15(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '14-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '15-2.pdf'), legend, 1e-7, 1e-2)

    legend=('a', 'b', 'c')
    

    


    # image_6_7(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '6-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '7-1.pdf'), legend, 1e-6, 2e-2)
    # image_6_7(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '6-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '7-2.pdf'), legend, 1e-6, 2e-2)

    # image_8_9(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '8-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '9-1.pdf'), legend, 1e-10, 1e-2)
    # image_8_9(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '8-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '9-2.pdf'), legend, 1e-10, 1e-2)

    # image_11_12(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '11-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '12-1.pdf'), legend, 1e-7, 2e-2)
    # image_11_12(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '11-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '12-2.pdf'), legend, 1e-7, 2e-2)

    # image_14_15(ks[0], os.path.join('images', 'fem_on_mvd_meshes', '14-1.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '15-1.pdf'), legend, 1e-7, 1e-2)
    # image_14_15(ks[1], os.path.join('images', 'fem_on_mvd_meshes', '14-2.pdf'), os.path.join('images', 'fem_on_mvd_meshes', '15-2.pdf'), legend, 1e-7, 1e-2)