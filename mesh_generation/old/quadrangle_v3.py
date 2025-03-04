import gmsh
import numpy as np


def center_circumscribed_circle(A, B, C):
    Ax, Ay, _ = A
    Bx, By, _ = B
    Cx, Cy, _ = C
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D
    return np.array((Ux, Uy, _))


def reverse_dict(d):
    res = {}
    for k, vs in d.items():
        for v in vs:
            if v not in res:
                res[v] = []
            res[v].append(k)
    return res


def generate_quadrangle_mesh(triangle_mesh, quadrangle_mesh):
    gmsh.initialize()

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    max_node_tag = gmsh.model.mesh.get_max_node_tag()
    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    # physical group = surface
    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]
        boundaries = gmsh.model.get_boundary(((dim, surf_tag),))
    
        triangles, triangles_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)

        triangle_to_center = {}
        centers, centers_coords = [], []
        for triangle, nodes in zip(triangles, triangles_nodes.reshape((-1, 3))):
            center_coords = center_circumscribed_circle(*map(lambda x: gmsh.model.mesh.get_node(x)[0], nodes))
            centers.append(max_node_tag + 1)
            centers_coords.append(center_coords)

            max_node_tag += 1
            triangle_to_center[triangle] = max_node_tag

        gmsh.model.mesh.add_nodes(dim, surf_tag, centers, np.array(centers_coords).flatten())

        edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)        
        edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
        triangle_to_edges = {triangle: edges for triangle, edges in zip(triangles, edge_tags.reshape(-1, 3))}
        edge_to_triangles = reverse_dict(triangle_to_edges)
        
        quads, quads_nodes = [], []
        boundary_to_tags_and_nodes = {boundary: ([], []) for boundary in boundaries}
        for edge, linked_triangles in edge_to_triangles.items():
            nodes = edge_to_nodes[edge]

            quads.append(max_element_tag + 1)

            if len(linked_triangles) == 2:
                quads_nodes += [nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], triangle_to_center[linked_triangles[1]]]
            else:
                boundary = None
                coords1, parametric_coords1, dim1, tag1 = gmsh.model.mesh.get_node(nodes[0])
                coords2, parametric_coords2, dim2, tag2 = gmsh.model.mesh.get_node(nodes[1])
                if dim1 == 1:
                    boundary = dim1, tag1
                elif dim2 == 1:
                    boundary = dim2, tag2
                else:   # dim1 = dim2 = 0
                    upward1, downward1 = gmsh.model.get_adjacencies(dim1, tag1)
                    upward2, downward2 = gmsh.model.get_adjacencies(dim2, tag2)
                    boundary = 1, set(upward1).intersection(upward2).pop()

                edge_center = (coords1 + coords2) / 2

                boundary_to_tags_and_nodes[boundary][0].append(max_node_tag + 1)
                boundary_to_tags_and_nodes[boundary][1].append(edge_center)

                max_node_tag += 1
                quads_nodes += [nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], max_node_tag]

            max_element_tag += 1

        for boundary in boundaries:
            gmsh.model.mesh.add_nodes(*boundary, boundary_to_tags_and_nodes[boundary][0], np.array(boundary_to_tags_and_nodes[boundary][1]).flatten())

        gmsh.model.mesh.add_elements_by_type(surf_tag, quadrangle_type, quads, quads_nodes)
        gmsh.model.mesh.remove_elements(dim, surf_tag, triangles)

    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(quadrangle_mesh)

    gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate_quadrangle_mesh('meshes/new_triangle.msh', 'meshes/new_quadrangle.msh')
