import gmsh
import numpy as np
import os


def generate(mesh_size, fname=None, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.geo.add_point(1, 0, 0)
    gmsh.model.geo.add_point(1, 1, 0)
    gmsh.model.geo.add_point(0, 1, 0)
    gmsh.model.geo.add_point(0, 0, 0)

    gmsh.model.geo.add_line(1, 2)
    gmsh.model.geo.add_line(2, 3)
    gmsh.model.geo.add_line(3, 4)
    gmsh.model.geo.add_line(4, 1)

    gmsh.model.geo.add_curve_loop([1, 2, 3, 4])

    gmsh.model.geo.add_plane_surface([1])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [2])
    gmsh.model.addPhysicalGroup(2, [1])

    gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size)
    gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size)

    gmsh.model.mesh.generate(2)

    if fname is not None:
        gmsh.write(fname)

    if ui:
        gmsh.fltk.run()
    
    node_tags, coords, _ = gmsh.model.mesh.get_nodes()
    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    
    gmsh.finalize()

    return node_tags.size, element_tags.size


def run(basic_dir):
    ns = (128, 256, 512)
    mesh_sizes = 1 / np.sqrt(np.array(ns) / 2)

    for ms, n in zip(mesh_sizes, ns):
        n_nodes, n_cells = generate(ms, os.path.join(basic_dir, f'unit_square_{n}.msh'), False)
        print(((n + 1) ** 2), n*n*2, n_nodes, n_cells)
        #while n_nodes < ((n + 1) ** 2) * 0.9:
        while n_cells < n*n*2 * 0.8:
            n_nodes, n_cells = generate(ms, os.path.join(basic_dir, f'unit_square_{n}.msh'), False)
            print(((n + 1) ** 2), n*n*2, n_nodes, n_cells)
            ms *= 0.9
            #print((n+1)**2, n*n*2)
            #print(n_nodes, n_cells)
            #print()


if __name__ == '__main__':
    basic_dir = os.path.join('meshes', 'NS')
    os.makedirs(basic_dir, exist_ok=True)
    
    run(basic_dir)