import gmsh
import numpy as np
import utility


def generate_quadrangle_mesh(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    #max_node_tag = gmsh.model.mesh.get_max_node_tag()
    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]
        boundaries = gmsh.model.get_boundary(((dim, surf_tag),))
    
        triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
        center_tags = utility.add_triangle_centers(dim, surf_tag, triangle_nodes)
        triangle_to_center = {triangle: center for triangle, center in zip(triangle_tags, center_tags)}

        edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)        
        edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
        triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
        edge_to_triangles = utility.reverse_dict(triangle_to_edges)
        
        max_node_tag = gmsh.model.mesh.get_max_node_tag()
        quadrangle_tags, quadrangle_nodes = [], []
        boundary_triangle_tags, boundary_triangle_nodes = [], []
        boundary_to_nodes_and_coords = {boundary: ([], []) for boundary in boundaries}
        for edge, linked_triangles in edge_to_triangles.items():
            nodes = edge_to_nodes[edge]

            max_element_tag += 1
            

            if len(linked_triangles) == 2:
                quadrangle_tags.append(max_element_tag)
                quadrangle_nodes.extend((nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], triangle_to_center[linked_triangles[1]]))
            else:
                # coords1, parametric_coords1, dim1, tag1 = gmsh.model.mesh.get_node(nodes[0])
                # coords2, parametric_coords2, dim2, tag2 = gmsh.model.mesh.get_node(nodes[1])
                # boundary = utility.get_boundary(dim1, tag1, dim2, tag2)

                # edge_center = (coords1 + coords2) / 2

                # max_node_tag += 1
                # boundary_to_nodes_and_coords[boundary][0].append(max_node_tag)
                # boundary_to_nodes_and_coords[boundary][1].extend(edge_center)
                
                # quadrangle_nodes.extend((nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], max_node_tag))
                boundary_triangle_tags.append(max_element_tag)
                boundary_triangle_nodes.extend((nodes[0], triangle_to_center[linked_triangles[0]], nodes[1]))

        # for boundary in boundaries:
        #     gmsh.model.mesh.add_nodes(*boundary, boundary_to_nodes_and_coords[boundary][0], boundary_to_nodes_and_coords[boundary][1])

        gmsh.model.mesh.add_elements_by_type(surf_tag, quadrangle_type, quadrangle_tags, quadrangle_nodes)
        gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, boundary_triangle_tags, boundary_triangle_nodes)
        gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)

    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(fname)

    entities = gmsh.model.getEntities()
    for entity in entities:
        dim, tag = entity
        print(entity, gmsh.model.getType(dim, tag))   # , gmsh.model.getEntityName(dim, tag)
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        print(nodeTags, nodeCoords, nodeParams)
        print(elemTypes, elemTags, elemNodeTags)
    
    boundary_entities = gmsh.model.get_boundary([(2, 1)])
    print(boundary_entities)
    for boundary_entity in boundary_entities:
        print(boundary_entity, gmsh.model.mesh.get_nodes(*boundary_entity))

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate_quadrangle_mesh('meshes/triangle_subdomains.msh', 'meshes/new_quadrangle.msh', ui=True)
