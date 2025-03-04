import gmsh
import numpy as np


# пересечение медиан, центроид, барицентр
def generate(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

    max_element_tag = gmsh.model.mesh.get_max_element_tag()
    max_node_tag = gmsh.model.mesh.get_max_node_tag()

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            triangles_tags, triangles_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)

            triangle_barycenters = gmsh.model.mesh.get_barycenters(triangle_type, surf_tag, fast=False, primary=True)

            new_nodes = []
            small_triangle_tags, small_triangle_nodes = [], []
            for triangle_nodes, barycenter_coords in zip(triangles_nodes.reshape((-1, 3)), triangle_barycenters.reshape(-1, 3)):
                max_node_tag += 1
                new_nodes.append(max_node_tag)

                max_element_tag += 3
                small_triangle_tags.extend((max_element_tag - 2, max_element_tag - 1, max_element_tag))

                for node1, node2 in zip(triangle_nodes, np.roll(triangle_nodes, -1)):
                    small_triangle_nodes.extend((node1, node2, max_node_tag))

            
            gmsh.model.mesh.add_nodes(dim, surf_tag, new_nodes, triangle_barycenters)

            gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, small_triangle_tags, small_triangle_nodes)
            gmsh.model.mesh.remove_elements(dim, surf_tag, triangles_tags)

    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(fname)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate('meshes/msh/rectangle_0_triangle_optimized.msh', 'meshes/msh/rectangle_1_optimized.msh', True)
