import gmsh
import small_quadrangle
import utility
import numpy as np


# boundary problems
def refine(triangle_mesh, refined_mesh, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    # if ui:
    #     gmsh.fltk.run()

    gmsh.model.mesh.refine()
    
    if ui:
        gmsh.fltk.run()
    
    gmsh.write(refined_mesh)

    gmsh.finalize()


def uniform_split(triangle_mesh, refined_mesh, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)

            edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)        
            edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
            triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
            edge_to_triangles = utility.reverse_dict(triangle_to_edges)
            edge_to_center = small_quadrangle.add_edge_centers(dim, surf_tag, edge_to_triangles, edge_to_nodes)

            new_triangle_tags, new_triangle_nodes = [], []
            for triangle, edges, nodes in zip(triangle_tags, edge_tags.reshape(-1, 3), edge_nodes.reshape(-1, 6)):
                for edge1, edge2, node in zip(edges, np.roll(edges, -1), nodes[1::2]):
                    max_element_tag += 1
                    new_triangle_tags.append(max_element_tag)
                    new_triangle_nodes.extend((node, edge_to_center[edge1], edge_to_center[edge2]))
                max_element_tag += 1
                new_triangle_tags.append(max_element_tag)
                new_triangle_nodes.extend(map(lambda x: edge_to_center[x], edges))

            gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, new_triangle_tags, new_triangle_nodes)
            gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)

    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(refined_mesh)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    #refine('meshes/msh/rectangle_1_2_3_4_triangle_optimized.msh', 'meshes/msh/rectangle_1_optimized.msh', True)
    uniform_split('meshes/msh/rectangle_1_2_3_4_triangle_optimized.msh', 'meshes/msh/rectangle_1_optimized.msh', True)
