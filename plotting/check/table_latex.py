import gmsh
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('mesh_generation')
import utility



gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
for i in range(22):
    gmsh.open(f'meshes/rectangle/rectangle_{i}_quadrangle.msh')

    quadrangle_tags, quadrangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))

    loaded = np.load(f'meshes/rectangle/rectangle_{i}_quadrangle.npz', allow_pickle=True)
    node_groups = loaded['node_groups'].astype(int)
    cell_nodes = loaded['cells'] - 1

    number_of_D_nodes = node_groups[0] + node_groups[2] - node_groups[1]
    number_of_V_nodes = node_groups[1] - node_groups[0] + node_groups[3] - node_groups[2]

    gmsh.open(f'meshes/rectangle/rectangle_{i}_triangle.msh')
    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    triangle_nodes -= 1
    triangle_nodes = triangle_nodes.reshape(-1, 3)
    nodes, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)
    node_coords = node_coords[:, :2]

    angles_1, angles_2 = [], []
    for nodes in node_coords[triangle_nodes]:
        angles_1.extend(utility.get_angles(nodes))
        angles_2.extend(utility.get_triangle_angles(nodes))
    angles_1 = np.array(angles_1)
    angles_2 = np.array(angles_2)
    
    assert np.allclose(angles_1, angles_2)

    print(rf'{i+1} & {number_of_D_nodes} & {angles_1.min():.1f} / {angles_1.max():.1f} & {number_of_V_nodes} & {quadrangle_tags.size} \\ \hline')

gmsh.finalize()

