import gmsh
import numpy as np


# get boundary problems
def split(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    if ui:
        gmsh.fltk.run()

    gmsh.model.mesh.splitQuadrangles()
    
    if ui:
        gmsh.fltk.run()
    
    gmsh.write(fname)

    gmsh.finalize()


def my_split(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    if ui:
        gmsh.fltk.run()

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            quad_tags, quads_nodes = gmsh.model.mesh.get_elements_by_type(quadrangle_type, surf_tag)

            triangle_tags, triangle_nodes = [], []
            for quad_nodes in quads_nodes.reshape(-1, 4):
                quad_coords = [node_coords[node - one] for node in quad_nodes]

                first_diagonal_len = np.linalg.norm(quad_coords[2] - quad_coords[0])
                second_diagonal_len = np.linalg.norm(quad_coords[3] - quad_coords[1])

                if first_diagonal_len < second_diagonal_len:
                    triangle_nodes.extend((*quad_nodes[:3], quad_nodes[0], *quad_nodes[2:]))    # 0, 1, 2 ; 0, 2, 3
                else:
                    triangle_nodes.extend((*quad_nodes[1:], *quad_nodes[:2], quad_nodes[3]))    # 1, 2, 3 ; 0, 1, 3
                
                if first_diagonal_len == second_diagonal_len:
                    print(first_diagonal_len, second_diagonal_len)
                
                triangle_tags.extend((max_element_tag + 1, max_element_tag + 2))

                max_element_tag += 2

            gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, triangle_tags, triangle_nodes)
            gmsh.model.mesh.remove_elements(dim, surf_tag, quad_tags)

    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()
    
    if ui:
        gmsh.fltk.run()
    
    gmsh.write(fname)

    gmsh.finalize()


if __name__ == '__main__':
    #refine('meshes/msh/rectangle_1_2_3_4_triangle_optimized.msh', 'meshes/msh/rectangle_1_optimized.msh', True)
    my_split('meshes/msh/rectangle_3_quadrangle.msh', 'meshes/msh/rectangle_1_optimized.msh', True)
