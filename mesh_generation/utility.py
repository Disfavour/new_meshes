import gmsh
import numpy as np


def reverse_dict(d):
    res = {}
    for k, vs in d.items():
        for v in vs:
            if v not in res:
                res[v] = []
            res[v].append(k)
    return res


def circumcenter(A, B, C):
    Ax, Ay, _ = A
    Bx, By, _ = B
    Cx, Cy, _ = C
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D
    return np.array((Ux, Uy, _))


def add_triangle_centers(dim, tag, triangles_nodes):
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    max_node_tag = gmsh.model.mesh.get_max_node_tag()
    center_tags, center_coords = [], []
    for nodes in triangles_nodes.reshape((-1, 3)):
        center_coord = circumcenter(*(node_coords[node - one] for node in nodes))

        max_node_tag += 1
        center_tags.append(max_node_tag)
        center_coords.extend(center_coord)

    gmsh.model.mesh.add_nodes(dim, tag, center_tags, center_coords)
    return center_tags


def get_boundary_of_nodes(node1, node2):
    coords1, parametric_coords1, dim1, tag1 = gmsh.model.mesh.get_node(node1)
    coords2, parametric_coords2, dim2, tag2 = gmsh.model.mesh.get_node(node2)
    return get_boundary(dim1, tag1, dim2, tag2)


def get_boundary(dim1, tag1, dim2, tag2):
    if dim1 == 1:
        return dim1, tag1
    elif dim2 == 1:
        return dim2, tag2
    else:   # dim1 = dim2 = 0
        upward1, downward1 = gmsh.model.get_adjacencies(dim1, tag1)
        upward2, downward2 = gmsh.model.get_adjacencies(dim2, tag2)
        return 1, set(upward1).intersection(upward2).pop()


def get_max_angle():
    angles = get_all_angles()
    return np.max(angles)


def get_all_angles():
    element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()
    node_tags = node_tags[0]
    node_tags = node_tags.reshape(-1, 3) if element_types == gmsh.model.mesh.get_element_type("Triangle", 1) else node_tags.reshape(-1, 4)
    angles = []
    for nodes in node_tags:
        node_coords = []
        for node in nodes:
            coords, parametric_coords, dim, tag = gmsh.model.mesh.get_node(node)
            node_coords.append(coords)
        angles.append(get_angles(node_coords))
    angles = np.array(angles)
    return angles


# угол между векторами
def get_angles(nodes):
    nodes = np.array(nodes)
    angles = []
    for a, b in zip(np.roll(nodes, 1, axis=0) - nodes, np.roll(nodes, -1, axis=0) - nodes):
        angle = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))))
        angles.append(angle)
    return angles


# теорема косинусов
def get_triangle_angles(nodes):
    nodes = np.array(nodes)
    lenghts = np.linalg.norm(nodes - np.roll(nodes, -1, axis=0), axis=1)
    angles = []
    for a, b, c in zip(np.roll(lenghts, -1), lenghts, np.roll(lenghts, 1)):
        angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2*b*c)))
        angles.append(angle)
    return angles
