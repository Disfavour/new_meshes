import gmsh
import numpy as np


def center_circumscribed_circle(A, B, C):
    Ax, Ay, _ = A
    Bx, By, _ = B
    Cx, Cy, _ = C
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D

    print(Ux, Uy, _, '->', (A+B+C)/3, ((A+B)/2 + (B+C)/2 + (A+C)/2)/3)
    return np.array((Ux, Uy, _))


def reverse_dict(d):
    res = {}
    for k, vs in d.items():
        for v in vs:
            if v not in res:
                res[v] = []
            res[v].append(k)
    return res


# для одной физической области
def generate_quadrangle_mesh(triangle_mesh, fname):
    gmsh.initialize()

    x1, y1, x2, y2 = -10, -10, 10, 10

    point1 = gmsh.model.geo.add_point(x1, y1, 0)
    point2 = gmsh.model.geo.add_point(x1, y2, 0)
    point3 = gmsh.model.geo.add_point(x2, y2, 0)
    point4 = gmsh.model.geo.add_point(x2, y1, 0)

    line1 = gmsh.model.geo.add_line(point1, point2)
    line2 = gmsh.model.geo.add_line(point2, point3)
    line3 = gmsh.model.geo.add_line(point3, point4)
    line4 = gmsh.model.geo.add_line(point4, point1)

    face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])

    gmsh.model.geo.add_plane_surface([face1])

    physgroup = gmsh.model.addPhysicalGroup(2, [1])

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", 10)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

    gmsh.model.mesh.generate(2)

    gmsh.write('meshes/new_triangle.msh')

    #gmsh.fltk.run()
    #gmsh.open(triangle_mesh)

    #gmsh.model.mesh.reclassifyNodes()
    #gmsh.model.mesh.createGeometry()
    #gmsh.model.mesh.classify_surfaces()
    
    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    triangles, triangles_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    #node_tags, node_coords, node_params = gmsh.model.mesh.get_nodes()

    triangle_to_center = {}
    max_node_tag = gmsh.model.mesh.get_max_node_tag()
    centers, centers_coords = [], []
    for triangle, nodes in zip(triangles, triangles_nodes.reshape((-1, 3))):
        center_coords = center_circumscribed_circle(*map(lambda x: gmsh.model.mesh.get_node(x)[0], nodes))
        centers.append(max_node_tag + 1)
        centers_coords.append(center_coords)

        max_node_tag += 1
        triangle_to_center[triangle] = max_node_tag

    gmsh.model.mesh.add_nodes(2, 1, centers, np.array(centers_coords).flatten())

    edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type)
    gmsh.model.mesh.createEdges()
    edge_tags, edge_orientations = gmsh.model.mesh.getEdges(edge_nodes)

    triangle_to_edges = {triangle: edges for triangle, edges in zip(triangles, edge_tags.reshape(-1, 3))}
    edge_to_triangles = reverse_dict(triangle_to_edges)

    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    max_element_tag = gmsh.model.mesh.get_max_element_tag()
    quads, quads_nodes = [], []
    boundary_nodes, boundary_nodes_coords = [], []
    boundary = gmsh.model.get_boundary(((2, 1),))

    for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2)):
        linked_triangles = edge_to_triangles[edge]

        quads.append(max_element_tag + 1)

        if len(linked_triangles) == 2:
            quads_nodes += [nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], triangle_to_center[linked_triangles[1]]]
        else:
            nodes_coords = np.array(tuple(map(lambda x: gmsh.model.mesh.get_node(x)[0], nodes))).flatten()
            
            for dim, tag in boundary:
                if gmsh.model.is_inside(dim, tag, nodes_coords) == 2:
                    break
            
            closest_coord, parametric_coord = gmsh.model.get_closest_point(dim, tag, gmsh.model.mesh.get_node(triangle_to_center[linked_triangles[0]])[0])
            boundary_nodes.append(max_node_tag + 1)
            boundary_nodes_coords.append(closest_coord)

            # лучше потом за раз добавить
            gmsh.model.mesh.add_nodes(dim, tag, [max_node_tag + 1], closest_coord)

            max_node_tag += 1
            quads_nodes += [nodes[0], triangle_to_center[linked_triangles[0]], nodes[1], max_node_tag]

        max_element_tag += 1
    
    quad_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)
    gmsh.model.mesh.add_elements_by_type(1, quad_type, quads, quads_nodes)
    gmsh.model.mesh.remove_elements(2, 1, triangles)

    #gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write('meshes/new_quad.msh')

    #gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate_quadrangle_mesh('experiment_triangle.msh', 'experiment_quadrangle.msh')
