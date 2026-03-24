import gmsh
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('mesh_generation')
import utility



gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
for i in range(22):
    gmsh.open(f'meshes/rectangle/rectangle_{i}_triangle.msh')
    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    triangle_nodes = triangle_nodes.reshape(-1, 3) - 1
    nodes, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)

    angles_1, angles_2 = [], []
    for nodes in node_coords[triangle_nodes]:
        angles_1.extend(utility.get_angles(nodes))
        angles_2.extend(utility.get_triangle_angles(nodes))
    angles_1 = np.array(angles_1)
    angles_2 = np.array(angles_2)
    
    assert np.allclose(angles_1, angles_2)

    print(rf'{i+1} & {node_coords.shape[0]} & {triangle_tags.size} & {angles_1.min():.2f} / {angles_1.max():.2f} \\')

gmsh.finalize()

