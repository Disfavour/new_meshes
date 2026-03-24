import gmsh
import utility
import quadrangle
import basic
import os
import meshio
#import optimization.optimize
import sys
sys.path.append('mesh_generation/refinement')
import circumcenter
import circumcenter_6
import split_quadrangles
import small_quadrangle
import time
import numpy as np


def compute_max_angle():
    nodes, coords, parametric_coords = gmsh.model.mesh.get_nodes()
    element_tags, elements = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    coords = coords.reshape(-1, 3)
    elements = elements.reshape(-1, 3) - 1

    assert nodes.size == nodes.max()
    if not np.all(nodes[:-1] < nodes[1:]):
        indices = np.argsort(nodes)
        nodes = nodes[indices]
        coords = coords[indices]
    assert np.all(nodes[:-1] < nodes[1:])

    return utility.compute_angles(coords[elements]).max()


def generate_mesh_on_ellipse(mesh_size, max_angle, previous_elements_count, fname=None, ui=False):
    ts = [time.time()]
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)
    
    height = 0.75
    width = 1

    gmsh.model.geo.add_point(0, height/2, 0)
    gmsh.model.geo.add_point(width/2, 0, 0)
    gmsh.model.geo.add_point(width, height/2, 0)
    gmsh.model.geo.add_point(width/2, height, 0)

    gmsh.model.geo.add_point(width/2, height/2, 0)

    gmsh.model.geo.add_ellipse_arc(1, 5, 3, 2)
    gmsh.model.geo.add_ellipse_arc(2, 5, 3, 3)
    gmsh.model.geo.add_ellipse_arc(3, 5, 3, 4)
    gmsh.model.geo.add_ellipse_arc(4, 5, 3, 1)

    gmsh.model.geo.add_curve_loop([1, 2, 3, 4])

    gmsh.model.geo.add_plane_surface([1])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [3, 4])
    gmsh.model.addPhysicalGroup(1, [1, 2])

    gmsh.model.addPhysicalGroup(2, [1])

    gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size)
    gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size)

    #gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal для качества углов
    #gmsh.option.setNumber("Mesh.Optimize", 2)   # Сглаживание
    #gmsh.option.setNumber("Mesh.Optimize", 5)

    #gmsh.model.mesh.set_algorithm(2, 1, 6)

    ts.append(time.time())
    gmsh.model.mesh.generate(2)
    # assert utility.is_all_counter_clockwise()
    ts.append(time.time())
    generate_time = ts[-1] - ts[-2]

    max_angle_generate = compute_max_angle()

    ts.append(time.time())
    gmsh.model.mesh.optimize('Relocate2D', niter=20)
    ts.append(time.time())
    optimize_time = ts[-1] - ts[-2]

    max_angle_optimize = compute_max_angle()
    nodes, coords, parametric_coords = gmsh.model.mesh.get_nodes()
    element_tags, elements = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))

    success = element_tags.size != previous_elements_count and max_angle_optimize < max_angle

    ts.append(time.time())
    if fname is not None and success:
        gmsh.write(fname)
    time_write = time.time() - ts[-1]

    if ui:
        gmsh.fltk.run()
        
    gmsh.finalize()

    full_time = ts[-1] - ts[0]
    other_time = full_time - (generate_time + optimize_time)
    print(f'{nodes.size:7d} {element_tags.size:7d} optimize {max_angle_generate:6.2f} -> {max_angle_optimize:6.2f} time {full_time:6.2f} generate {generate_time/full_time:6.2%} optimize {optimize_time/full_time:6.2%} other {other_time/full_time:6.2%}')
    print(f'write mesh {time_write:6.2f}')

    return success, element_tags.size


def generate_set_of_meshes(basic_dir, n=10):
    mesh_size = 0.5   # 0.2
    prevoius_element_count = 0
    for i in range(1, n+1):
        while True:
            success, element_count = generate_mesh_on_ellipse(mesh_size, 88, prevoius_element_count, os.path.join(basic_dir, f'triangle_{i}.msh'))

            if success:
                start_time = time.time()
                quadrangle.generate(os.path.join(basic_dir, f'triangle_{i}.msh'), os.path.join(basic_dir, f'quadrangle_{i}'))
                print(f'{i}/{n} generate quadrangle mesh {time.time() - start_time:6.2f}')

                prevoius_element_count = element_count
                break

            mesh_size *= 0.999
        
        mesh_size /= 2


if __name__ == '__main__':
    basic_dir = os.path.join('meshes', 'ellipse')
    os.makedirs(basic_dir, exist_ok=True)

    generate_set_of_meshes(basic_dir, n=10)

    # 81.19786204503613 20 0.49483865960020695

    # S = 1 * 0.75 = 0.75
    
    #experiment_rectangle(basic_dir, n=10)

    
    #create_other_meshes(basic_dir, n=22)
    
    #generate_mesh_on_rectangle(0.2, ui=True)

    # i = 10
    # start_time = time.time()
    # quadrangle.generate(os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'), os.path.join(basic_dir, f'rectangle_{i}_quadrangle'))
    # print(f'{i}/{10} {time.time() - start_time:6.2f}')

    # (1, 80.85376411884494) (2, 82.14206865858054) (3, 81.05212805711211) (4, 82.70382275973724) (5, 82.18549683171656)
    # (6, 82.41787571564656) (7, 82.84940263253095) (8, 83.1590876283211) (9, 83.40532737493766) (10, 84.84716528704078)
    