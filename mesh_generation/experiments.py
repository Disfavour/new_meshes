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


def generate_mesh_on_rectangle(mesh_size, fname=None, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.geo.add_point(1, 0, 0)
    gmsh.model.geo.add_point(1, 0.75, 0)
    gmsh.model.geo.add_point(0, 0.75, 0)
    gmsh.model.geo.add_point(0, 0, 0)

    gmsh.model.geo.add_line(1, 2)
    gmsh.model.geo.add_line(2, 3)
    gmsh.model.geo.add_line(3, 4)
    gmsh.model.geo.add_line(4, 1)

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

    ts = [time.time()]
    gmsh.model.mesh.generate(2)
    ts.append(time.time())

    # assert utility.is_all_counter_clockwise()
    # ts.append(time.time())

    nodes, coords, parametric_coords = gmsh.model.mesh.get_nodes()
    element_tags, elements = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    coords = coords.reshape(-1, 3)
    elements = elements.reshape(-1, 3) - 1

    print(f'before optimization {utility.compute_angles(coords[elements]).max():6.2f}')

    start_time = time.time()
    # gmsh.model.mesh.optimize('Laplace2D', niter=2)
    gmsh.model.mesh.optimize('Relocate2D')
    print(f'optimize {time.time() - start_time:6.2f}')

    ts[-1] = time.time()

    if fname is not None:
        gmsh.write(fname)

    if ui:
        gmsh.fltk.run()
    
    ts.append(time.time())
    
    nodes, coords, parametric_coords = gmsh.model.mesh.get_nodes()
    element_tags, elements = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    coords = coords.reshape(-1, 3)
    elements = elements.reshape(-1, 3) - 1

    assert nodes.size == nodes.max()
    # if not np.all(nodes[:-1] < nodes[1:]):
    #     indices = np.argsort(nodes)
    #     nodes = nodes[indices]
    #     node_coords = node_coords[indices]
    assert np.all(nodes[:-1] < nodes[1:])

    angles = utility.compute_angles(coords[elements])
    ts.append(time.time())

    max_angle = angles.max()

    full = ts[-1] - ts[0]
    generate = ts[1] - ts[0]
    write = ts[2] - ts[1]
    angles = ts[3] - ts[2]
    print(f'{max_angle:6.2f} {nodes.size:7d} {element_tags.size:7d} {mesh_size:4.2e} time {full:5.2f} (generate {generate:5.2f} ({generate/full:6.2%}) write {write:5.2f} ({write/full:6.2%}) angles {angles:5.2f} ({angles/full:6.2%}))')
    
    gmsh.finalize()

    return max_angle, nodes.size, element_tags.size


def experiment_rectangle(basic_dir, n=10):
    angles = []
    attmepts = []
    mesh_size = 0.49   # 0.2
    prevoius_number_of_elements = 0
    for i in range(1, n+1):
        attmepts.append(0)
        delta = mesh_size / 100
        while True:
            #start_time = time.time()
            max_angle, nodes_number, elements_number = generate_mesh_on_rectangle(mesh_size, os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'))
            #print(f'{max_angle:6.2f} {nodes_number:7d} {elements_number:7d} {mesh_size:4.2e} {time.time() - start_time:6.2f}')
            attmepts[-1] += 1

            if max_angle < 85 and elements_number > prevoius_number_of_elements:
                start_time = time.time()
                quadrangle.generate(os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'), os.path.join(basic_dir, f'rectangle_{i}_quadrangle'))
                print(f'{i}/{n} {time.time() - start_time:6.2f}')

                prevoius_number_of_elements = elements_number
                angles.append(max_angle)
                break

            mesh_size *= 0.999
        
        # mesh_size /= 1.5 ** 0.5
        mesh_size /= 2
        #mesh_size *= 1.01 / 2
    
    print(*enumerate(angles, 1))


def create_other_meshes(basic_dir, n):
    for i in range(n):
        triangle_mesh = os.path.join(basic_dir, f'rectangle_{i}_triangle.msh')
        quadrangle_mesh = os.path.join(basic_dir, f'rectangle_{i}_quadrangle.msh')
        circumcenter.generate(triangle_mesh, os.path.join(basic_dir, f'rectangle_{i}_triangle_circumcenter_3.msh'))
        circumcenter_6.generate(triangle_mesh, os.path.join(basic_dir, f'rectangle_{i}_triangle_circumcenter_6.msh'))
        split_quadrangles.split(quadrangle_mesh, os.path.join(basic_dir, f'rectangle_{i}_quadrangle_split.msh'))
        small_quadrangle.generate(triangle_mesh, os.path.join(basic_dir, f'rectangle_{i}_small_quadrangle.msh'))


def get_mesh_size_from_number_of_cells(N=1e5):
    S = 1 * 0.75
    mesh_size = (S / N * 4 / 3 ** 0.5) ** 0.5
    return mesh_size


# mesh_size / 2 -> x4 elements

if __name__ == '__main__':
    basic_dir = os.path.join('meshes', 'rectangle1')
    os.makedirs(basic_dir, exist_ok=True)

    # 81.19786204503613 20 0.49483865960020695

    # S = 1 * 0.75 = 0.75
    
    experiment_rectangle(basic_dir, n=10)
    #create_other_meshes(basic_dir, n=22)
    
    #generate_mesh_on_rectangle(0.2, ui=True)

    # i = 10
    # start_time = time.time()
    # quadrangle.generate(os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'), os.path.join(basic_dir, f'rectangle_{i}_quadrangle'))
    # print(f'{i}/{10} {time.time() - start_time:6.2f}')

    # (1, 80.85376411884494) (2, 82.14206865858054) (3, 81.05212805711211) (4, 82.70382275973724) (5, 82.18549683171656)
    # (6, 82.41787571564656) (7, 82.84940263253095) (8, 83.1590876283211) (9, 83.40532737493766) (10, 84.84716528704078)
    