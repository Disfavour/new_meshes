import gmsh
import numpy as np
import os
import time
import sys
sys.path.append('mesh_generation')
import utility


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


def generate(mesh_size, fname=None, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.geo.add_point(0, 0.25, 0)
    gmsh.model.geo.add_point(0.5, 0.25, 0)
    gmsh.model.geo.add_point(1, 0.25, 0)
    gmsh.model.geo.add_point(0.5, 0, 0)
    
    gmsh.model.geo.add_ellipse_arc(1, 2, 3, 4)
    gmsh.model.geo.add_ellipse_arc(4, 2, 3, 3)
    gmsh.model.geo.add_line(3, 1)

    gmsh.model.geo.add_curve_loop([1, 2, 3])

    gmsh.model.geo.add_plane_surface([1])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [3])
    gmsh.model.addPhysicalGroup(2, [1])

    gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size)
    gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size)

    start_time = time.time()
    gmsh.model.mesh.generate(2)
    generate_time = time.time() - start_time

    max_angle_generate = compute_max_angle()

    start_time = time.time()
    gmsh.model.mesh.optimize('Relocate2D', niter=20)
    optimize_time = time.time() - start_time

    max_angle_optimize = compute_max_angle()

    if fname is not None:
        gmsh.write(fname)

    if ui:
        gmsh.fltk.run()
    
    node_tags, coords, _ = gmsh.model.mesh.get_nodes()
    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    
    gmsh.finalize()

    return node_tags.size, element_tags.size


def run(fname):
    nodes_number = 801 ** 2 # 641601
    mesh_size = 0.0006461081889226679

    while True:
        n_nodes, n_cells = generate(mesh_size, fname)
        print(n_nodes, n_cells, nodes_number, mesh_size)

        if n_nodes > 5e5:
            break

        mesh_size *= 0.9


if __name__ == '__main__':
    basic_dir = os.path.join('meshes', 'NS')
    os.makedirs(basic_dir, exist_ok=True)
    
    #run(os.path.join(basic_dir, 'ellipse.msh'))

    #generate(0.0006461081889226679, os.path.join(basic_dir, 'ellipse.msh'))
    
    n_nodes, n_cells = generate(0.00135, os.path.join(basic_dir, 'ellipse_3.msh'))
    print(n_nodes, n_cells)

    # 'ellipse_128.msh' 129 ** 2 = 16641 / 0.00375 16547 32499
    # 'ellipse_256.msh' 257 ** 2 = 66049 / 0.00186 66367 131540
    # 'ellipse.msh' 405257 807560 0.00075
    # 'ellipse_3.msh' 125517 249391 0.00135

    # 19095 37551 641601 0.003486784401000002
    # 23466 46223 641601 0.003138105960900002
    # 29071 57353 641601 0.0028242953648100018
    # 35779 70682 641601 0.0025418658283290017
    # 44053 87134 641601 0.0022876792454961017
    # 54490 107900 641601 0.0020589113209464917
    # 66855 132512 641601 0.0018530201888518425
    # 82760 164188 641601 0.0016677181699666583
    # 101797 202115 641601 0.0015009463529699924
    # 125517 249391 641601 0.0013508517176729932
    # 155127 308429 641601 0.001215766545905694
    # 190970 379914 641601 0.0010941898913151245
    # 235321 468392 641601 0.0009847709021836122
    # 290776 579051 641601 0.0008862938119652509
    # 358994 715210 641601 0.0007976644307687258
    # 442177 881269 641601 0.0007178979876918532
    # 546174 1088920 641601 0.0006461081889226679
    # 673627 1343446 641601 0.0005814973700304011