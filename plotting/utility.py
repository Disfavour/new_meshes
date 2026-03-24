import numpy as np
import gmsh


def get_default_figsize(text_width=16.5):
    cm = 1 / 2.54
    width = text_width * cm
    height = 9 / 16 * width
    return width, height


def get_default_figsize_column(text_width=16.5):
    width, height = get_default_figsize(text_width)
    return 0.49*width, height


def get_figsize(x_len, y_len, text_width=16.5):
    width, height = get_default_figsize(text_width)
    if x_len / y_len >= 16 / 9:
        return (width, width / (x_len / y_len))
    else:
        return (x_len / y_len * height, height)
    



def get_text_width():
    cm = 1 / 2.54
    text_width = 16.5 * cm # 16.5 15.5
    return text_width


def get_default_height():
    return 9 / 16 * get_text_width()



    

def get_figsize_2_columns_default():
    text_width = get_text_width() * 0.49 # get_text_width() / 2
    return (text_width, 3 / 4 * text_width)


# def get_figsize_2_columns_default():
#     figsize = get_default_figsize()
#     return (figsize[0] / 2, figsize[1])


def get_figsize_2_columns(xlen, ylen):
    figsize = get_default_figsize()
    figsize = (figsize[0] / 2, figsize[1])
    if xlen / ylen >= figsize[0] / figsize[1]:
        return (figsize[0], figsize[0] / (xlen / ylen))
    else:
        return (xlen / ylen * figsize[1], figsize[1])


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


def get_intersection_point_of_lines(a1, a2, b1, b2):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return p_x, p_y

def compute_intersection_points_v2(x, y):
    x1, x3, x2, x4 = x
    y1, y3, y2, y4 = y
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return np.column_stack((p_x, p_y))


def read_quad_mesh(quadrangle_mesh):
    gmsh.initialize()
    gmsh.open(f'{quadrangle_mesh}.msh')

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(-1, -1)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    loaded = np.load(f'{quadrangle_mesh}.npz', allow_pickle=True)
    node_groups = loaded['node_groups']
    cell_nodes = loaded['cells'] - 1

    voronoi_cells = np.concatenate((cell_nodes[:node_groups[0]], cell_nodes[node_groups[1]:node_groups[2]]))
    voronoi_nodes = np.unique(np.concatenate(voronoi_cells, axis=None))
    voronoi_nodes_coords = node_coords[voronoi_nodes]
    voronoi_cells_coords = [node_coords[cell] for cell in voronoi_cells]

    triangle_cells = np.stack((*cell_nodes[node_groups[0]:node_groups[1]], *cell_nodes[node_groups[2]:]))
    triangle_nodes = np.unique(np.concatenate(triangle_cells, axis=None))
    triangle_nodes_coords = node_coords[triangle_nodes]
    triangle_cells_coords = node_coords[triangle_cells]

    return get_figsize(xmax - xmin, ymax - ymin), node_coords, cell_nodes, voronoi_cells, voronoi_cells_coords, voronoi_nodes_coords, triangle_cells, triangle_cells_coords, triangle_nodes_coords