import gmsh
import numpy as np
import utility


def add_edge_centers(dim, tag, edge_to_triangles, edge_to_nodes):
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    boundaries = gmsh.model.get_boundary(((dim, tag),), oriented=False)
    max_node_tag = gmsh.model.mesh.get_max_node_tag()
    edge_to_center = {}
    edge_center_tags, edge_center_coords = [], []
    boundary_to_nodes_and_coords = {boundary: ([], []) for boundary in boundaries}
    for edge, linked_triangles in edge_to_triangles.items():
        nodes = edge_to_nodes[edge]
        nodes_coords = [node_coords[node - one] for node in nodes]
        edge_center_coord = (nodes_coords[0] + nodes_coords[1]) / 2

        max_node_tag += 1

        if len(linked_triangles) == 2:
            edge_center_tags.append(max_node_tag)
            edge_center_coords.extend(edge_center_coord)

        else:   # == 1
            boundary = utility.get_boundary_of_nodes(*nodes)

            boundary_to_nodes_and_coords[boundary][0].append(max_node_tag)
            boundary_to_nodes_and_coords[boundary][1].extend(edge_center_coord)
        
        edge_to_center[edge] = max_node_tag
        
    for boundary in boundaries:
        gmsh.model.mesh.add_nodes(*boundary, boundary_to_nodes_and_coords[boundary][0], boundary_to_nodes_and_coords[boundary][1])
    gmsh.model.mesh.add_nodes(dim, tag, edge_center_tags, edge_center_coords)
    return edge_to_center


def generate(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
            center_tags = utility.add_triangle_centers(dim, surf_tag, triangle_nodes)

            edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)        
            edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
            triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
            edge_to_triangles = utility.reverse_dict(triangle_to_edges)
            edge_to_center = add_edge_centers(dim, surf_tag, edge_to_triangles, edge_to_nodes)

            quadrangle_tags, quadrangle_nodes = [], []
            for edges, nodes, center in zip(edge_tags.reshape(-1, 3), edge_nodes.reshape(-1, 6), center_tags):
                for node, linked_edges in zip(nodes[1::2], zip(edges, np.roll(edges, -1))):
                    max_element_tag += 1

                    quadrangle_tags.append(max_element_tag)
                    quadrangle_nodes.extend((node, edge_to_center[linked_edges[0]], center, edge_to_center[linked_edges[1]]))

            gmsh.model.mesh.add_elements_by_type(surf_tag, quadrangle_type, quadrangle_tags, quadrangle_nodes)
            gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)

    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(fname)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate('meshes/triangle.msh', 'meshes/new_small_quadrangle.msh', ui=True)
