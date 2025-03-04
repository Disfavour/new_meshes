import os
import os.path
import basic
import optimization
import quadrangle
import mesh_generation.refinement.circumcenter as circumcenter
import small_quadrangle
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np


def make_images(mesh, fname):
    mesh = meshio.read(mesh)
    points = mesh.points[:, :2]

    elements = mesh.cells_dict['quad'] if 'quad' in mesh.cells_dict else mesh.cells_dict['triangle']

    elements_points = [
        np.array([points[j] for j in i])
        for i in elements
    ]

    plt.figure(figsize=(6.4, 3.6), dpi=300, tight_layout=True)
    ax = plt.gca()
    pc = PolyCollection(elements_points, facecolors='white', edgecolors='black')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    plt.savefig(fname, transparent=True)
    plt.close()


def optimize_and_generate(mesh_name, basic_dir_for_meshes, basic_triangle_mesh, images_dir):
    optimized_triangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}_optimized.msh')
    quadrangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}_quadrangle.msh')
    small_triangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}_small_triangle.msh')
    small_quadrangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}_small_quadrangle.msh')

    start_max_angle, max_angle = optimization.optimize_max_angle(basic_triangle_mesh, optimized_triangle_mesh)
    print(f'Mesh "{mesh_name}" optimization {start_max_angle} ({basic_triangle_mesh}) -> {max_angle} ({optimized_triangle_mesh})')

    make_images(os.path.join(basic_dir_for_meshes, f'{mesh_name}.msh'), os.path.join(images_dir, f'{mesh_name}.pdf'))
    make_images(os.path.join(basic_dir_for_meshes, f'{mesh_name}_optimized.msh'), os.path.join(images_dir, f'{mesh_name}_optimized.pdf'))

    if max_angle < 90:
        quadrangle.generate(optimized_triangle_mesh, quadrangle_mesh)
        circumcenter.generate(optimized_triangle_mesh, small_triangle_mesh)
        small_quadrangle.generate(optimized_triangle_mesh, small_quadrangle_mesh)

        make_images(os.path.join(basic_dir_for_meshes, f'{mesh_name}_quadrangle.msh'), os.path.join(images_dir, f'{mesh_name}_quadrangle.pdf'))
        make_images(os.path.join(basic_dir_for_meshes, f'{mesh_name}_small_triangle.msh'), os.path.join(images_dir, f'{mesh_name}_small_triangle.pdf'))
        make_images(os.path.join(basic_dir_for_meshes, f'{mesh_name}_small_quadrangle.msh'), os.path.join(images_dir, f'{mesh_name}_small_quadrangle.pdf'))

        print(f'Meshes {quadrangle_mesh, small_triangle_mesh, small_quadrangle_mesh} have been generated')
    else:
        print(f'Meshes havent been generated because "{optimized_triangle_mesh}" has max angle {max_angle} > 90')
    print()


if __name__ == '__main__':
    basic_dir_for_meshes = 'meshes'
    if not os.path.isdir(basic_dir_for_meshes):
        os.mkdir(basic_dir_for_meshes)
    
    images_dir = 'images'
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)


    mesh_name = 'rectangle'
    basic_triangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}.msh')
    polygon_points = (
        (0, 0), (0, 0.75), (1, 0.75), (1, 0),
    )
    polygon_mesh_size = 0.2
    basic.polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, basic_triangle_mesh)
    optimize_and_generate(mesh_name, basic_dir_for_meshes, basic_triangle_mesh, images_dir)


    mesh_name = 'polygon_with_holes'
    basic_triangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}.msh')
    polygon_points = (
        (0, 0), (0, 1), (2, 1), (2, 0),
    )
    holes_points = (
        ((0.2, 0.2), (0.5, 0.8), (0.8, 0.2)),
        ((1.2, 0.2), (1.2, 0.8), (1.8, 0.8), (1.8, 0.2)),
    )
    polygon_mesh_size = 0.15
    holes_mesh_sizes = (0.15, 0.15)
    basic.polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, basic_triangle_mesh, holes_points, holes_mesh_sizes)
    optimize_and_generate(mesh_name, basic_dir_for_meshes, basic_triangle_mesh, images_dir)


    mesh_name = 'rectangle_with_subdomains'
    basic_triangle_mesh = os.path.join(basic_dir_for_meshes, f'{mesh_name}.msh')
    # Первое ребро общее у подобластей (mesh_size_shared для общего ребра)
    subdomain_1_points = (
        (0.4, 0),
        (0.6, 0.75),
        (0, 0.75),
        (0, 0),
    )
    subdomain_2_points = (
        (0.4, 0),
        (0.6, 0.75),
        (1, 0.75),
        (1, 0),
    )
    mesh_size = 0.2
    mesh_size_shared = 0.1 # для общего ребра
    basic.rectangle_with_subdomains(subdomain_1_points, subdomain_2_points, mesh_size, mesh_size_shared, basic_triangle_mesh)
    optimize_and_generate(mesh_name, basic_dir_for_meshes, basic_triangle_mesh, images_dir)
    