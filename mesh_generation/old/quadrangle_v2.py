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

    # x1, y1, x2, y2 = -10, -10, 10, 10

    # point1 = gmsh.model.geo.add_point(x1, y1, 0)
    # point2 = gmsh.model.geo.add_point(x1, y2, 0)
    # point3 = gmsh.model.geo.add_point(x2, y2, 0)
    # point4 = gmsh.model.geo.add_point(x2, y1, 0)

    # line1 = gmsh.model.geo.add_line(point1, point2)
    # line2 = gmsh.model.geo.add_line(point2, point3)
    # line3 = gmsh.model.geo.add_line(point3, point4)
    # line4 = gmsh.model.geo.add_line(point4, point1)

    # face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])

    # gmsh.model.geo.add_plane_surface([face1])

    # physgroup = gmsh.model.addPhysicalGroup(2, [1])

    # gmsh.model.geo.synchronize()

    x1=0
    y1=0
    x2=1
    y2=0.75
    # lx1=0.4
    # lx2=0.6
    lx1=0.4
    lx2=0.6
    # meshsize=0.3
    # meshsize_along_line=0.1
    meshsize= 0.02
    meshsize_along_line= 0.02 #/ 4
    p1 = gmsh.model.geo.add_point(x1, y1, 0, meshsize)
    p2 = gmsh.model.geo.add_point(x1, y2, 0, meshsize)
    p3 = gmsh.model.geo.add_point(x2, y2, 0, meshsize)
    p4 = gmsh.model.geo.add_point(x2, y1, 0, meshsize)
    p5 = gmsh.model.geo.add_point(lx1, y1, 0, meshsize_along_line)
    p6 = gmsh.model.geo.add_point(lx2, y2, 0, meshsize_along_line)

    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_line(p2, p6)
    l3 = gmsh.model.geo.add_line(p6, p5)
    l4 = gmsh.model.geo.add_line(p5, p1)

    l5 = gmsh.model.geo.add_line(p5, p4)
    l6 = gmsh.model.geo.add_line(p4, p3)
    l7 = gmsh.model.geo.add_line(p3, p6)

    face1 = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    face2 = gmsh.model.geo.add_curve_loop([l5, l6, l7, l3])

    gmsh.model.geo.add_plane_surface([face1])
    gmsh.model.geo.add_plane_surface([face2])

    gmsh.model.geo.synchronize()

    physgroup1 = gmsh.model.addPhysicalGroup(2, [1])
    physgroup1 = gmsh.model.addPhysicalGroup(2, [2])

    #gmsh.option.setNumber("Mesh.MeshSizeMin", 10)
    #gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

    gmsh.model.mesh.generate(2)

    gmsh.model.mesh.optimize(method='Relocate2D', niter=10000)

    gmsh.write(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quad_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

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
                nodes_coords = np.array(tuple(map(lambda x: gmsh.model.mesh.get_node(x)[0], nodes))).flatten()
                
                for boundary in boundaries:
                    if gmsh.model.is_inside(*boundary, nodes_coords) == 2:
                        break

                # Можно находить среднюю точку вместо высоты (свойства центра описанной окружности)
                closest_coord, parametric_coord = gmsh.model.get_closest_point(*boundary, gmsh.model.mesh.get_node(triangle_to_center[linked_triangles[0]])[0])

                boundary_to_tags_and_nodes[boundary][0].append(max_node_tag + 1)
                boundary_to_tags_and_nodes[boundary][1].append(closest_coord)

                max_node_tag += 1
                quads_nodes += [nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], max_node_tag]

            max_element_tag += 1

        for boundary in boundaries:
            gmsh.model.mesh.add_nodes(*boundary, boundary_to_tags_and_nodes[boundary][0], np.array(boundary_to_tags_and_nodes[boundary][1]).flatten())

        gmsh.model.mesh.add_elements_by_type(surf_tag, quad_type, quads, quads_nodes)
        gmsh.model.mesh.remove_elements(dim, surf_tag, triangles)

    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(quadrangle_mesh)

    gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate_quadrangle_mesh('meshes/new_triangle.msh', 'meshes/new_quad.msh')
