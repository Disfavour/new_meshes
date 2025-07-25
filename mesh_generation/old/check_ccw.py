import numpy as np


# Ориентированная площадь
def is_ccw(*points):
    area = 0
    for p1, p2 in zip(points, np.roll(points, -1, axis=0)):
        area += np.cross(p1, p2)[2]
    return area > 0  # True → CCW, False → CW


def sort_ccw(*points):
    centroid = np.sum(points, axis=0) / len(points)
    return sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))


if __name__ == '__main__':
    import gmsh
    gmsh.initialize()
    gmsh.open('meshes/msh/rectangle_1_triangle.msh')    # rectangle_1_triangle  rectangle_1_quadrangle

    

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()
    node_tags = node_tags[0]
    node_tags = node_tags.reshape(-1, 3) if element_types == gmsh.model.mesh.get_element_type("Triangle", 1) else node_tags.reshape(-1, 4)
    #element_tags = element_tags[0]
    #element_tags = element_tags.reshape(-1, 3) if element_types == gmsh.model.mesh.get_element_type("Triangle", 1) else element_tags.reshape(-1, 4)

    for nodes in node_tags:
        a = is_ccw(*map(lambda x: node_coords[x - one], nodes))
        print(a)
        #print(sort_ccw(*map(lambda x: node_coords[x - one], nodes)))
    
    print(node_tags)

    gmsh.fltk.run()
    

    gmsh.finalize()
