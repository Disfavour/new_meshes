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
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    assert np.all(node_tags[:-1] < node_tags[1:])
    assert node_tags.size == node_tags.max()
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
    return np.array(center_tags, dtype=np.uint64)


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

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    element_type = triangle_type if triangle_type in element_types else quadrangle_type
    
    node_tags = node_tags[np.argwhere(element_types == element_type)[0][0]]
    node_tags = node_tags.reshape(-1, 3) if element_type == triangle_type else node_tags.reshape(-1, 4)

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


def is_all_counter_clockwise():
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    assert node_tags.size == node_tags.max()
    # assert np.all(node_tags[:-1] < node_tags[1:])
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    
    assert np.all(node_tags[:-1] < node_tags[1:])

    element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    element_type = triangle_type if triangle_type in element_types else quadrangle_type
    
    node_tags = node_tags[np.argwhere(element_types == element_type)[0][0]]
    node_tags = node_tags.reshape(-1, 3) if element_type == triangle_type else node_tags.reshape(-1, 4)

    for nodes in node_tags:
        if not is_counter_clockwise([node_coords[node - one] for node in nodes]):
            return False
    
    return True


# Удвоенная знаковая (ориентированная) площадь треугольника
def signed_triangle_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)


# Проверка для треугольника, но (3 точки против часовой) -> (все точки против часовой) Нужно быть внимательным с треугольниками нулевой площади
def is_counter_clockwise2(points):
    area = signed_triangle_area(*(p[:2] for p in points[:3]))
    assert not np.isclose(area, 0)
    return area > 0


def is_counter_clockwise(points):
    points = np.asarray(points)
    centroid = np.sum(points, axis=0) / points.shape[0]
    area = signed_triangle_area(*([p[:2] for p in points[:2]] + [centroid[:2]]))
    assert not np.isclose(area, 0)
    return area > 0


def sort_counter_clockwise(nodes, points):
    points = np.asarray(points)
    centroid = np.sum(points, axis=0) / points.shape[0]
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    indices = np.argsort(angles)
    assert is_counter_clockwise(points[indices])    # потом убрать
    return np.asarray(nodes)[indices]
    #return sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))


# 2D
def get_intersection_point_of_lines(a1, a2, b1, b2):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return p_x, p_y


# Формула площади Гаусса (многоугольника)
def polygon_area(points):
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
