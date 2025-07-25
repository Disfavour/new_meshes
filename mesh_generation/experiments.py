import gmsh
import utility
import quadrangle
import basic
import os
#import optimization.optimize


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

    gmsh.model.mesh.generate(2)

    max_angle = utility.get_max_angle()

    assert utility.is_all_counter_clockwise()

    if fname is not None:
        gmsh.write(fname)

    if ui:
        gmsh.fltk.run()
    
    gmsh.finalize()

    return max_angle


def experiment_rectangle(basic_dir, n=10):
    mesh_size = 1   # 0.2
    for i in range(n):
        while True:
            max_angle = generate_mesh_on_rectangle(mesh_size, os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'))
            if max_angle < 85:
                print(max_angle)
                quadrangle.generate(os.path.join(basic_dir, f'rectangle_{i}_triangle.msh'), os.path.join(basic_dir, f'rectangle_{i}_quadrangle'))
                break

            mesh_size *= 0.99
        
        mesh_size /= 1.5 ** 0.5


if __name__ == '__main__':
    basic_dir = os.path.join('meshes', 'rectangle')
    os.makedirs(basic_dir, exist_ok=True)
    
    experiment_rectangle(basic_dir)
    
    #generate_mesh_on_rectangle(0.2, ui=True)
