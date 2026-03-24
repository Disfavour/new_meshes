import numpy as np
import gmsh


def compute_polygon_areas(x, y):
    '''Формула площади Гаусса (многоугольника)'''
    return np.abs(np.sum(x * np.roll(y, -1, axis=0), axis=0) - np.sum(y * np.roll(x, -1, axis=0), axis=0)) / 2


def compute_intersection_points(x, y):
    x1, x3, x2, x4 = x
    y1, y3, y2, y4 = y
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return np.column_stack((p_x, p_y))


def load_msh(fname):
    gmsh.initialize()
    gmsh.open(fname)
    node_tags, nodes, _ = gmsh.model.mesh.get_nodes()
    quadrangle_tags, quadrangles = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    gmsh.finalize()

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        nodes = nodes.reshape(-1, 3)[:, :2][np.argsort(node_tags)]
    
    quadrangles = quadrangles.reshape(-1, 4).astype(int) - 1
    return nodes, quadrangles


def load_npz(fname):
    with np.load(fname, allow_pickle=True) as npz:
        groups = npz['node_groups']
        cells = npz['cells']

    groups = np.insert(groups.astype(int), 0, 0)
    cells -= 1
    return groups, cells


def compute_cell_areas(cells, groups, nodes):
    # надо это в нпз как-то вставить или ноды перенумеровать, чтобы были структурированные масивы
    cell_areas = np.empty(groups[3])

    cells_V = cells[groups[1]:groups[2]].astype((int, (3,)), copy=False)
    areas_V = compute_polygon_areas(*nodes[cells_V].T)

    cells_D = cells[np.r_[:groups[1], groups[2]:groups[3]]]
    cell_sizes = np.fromiter((cell.size for cell in cells_D), dtype=int, count=cells_D.shape[0])
    areas_D = np.empty(groups[1] + groups[3] - groups[2])
    for number_of_nodes in np.unique(cell_sizes):
        indexes = cell_sizes == number_of_nodes
        cells_specified = cells_D[indexes].astype((int, (number_of_nodes,)), copy=False)
        areas_specified = compute_polygon_areas(*nodes[cells_specified].T)
        areas_D[indexes] = areas_specified
    cell_areas = np.concatenate((areas_D[:groups[1]], areas_V, areas_D[groups[1]:]))
    return cell_areas


def redirect_eV_on_boundary(quadrangles, nodes, boundary):
    # это надо сделать при генерации сетки
    # если не совпадает можно просто поменять половинки квадрангла (его ноды) [29 36 34  6] -> [34  6 29 36]
    # Делаем чтобы нормаль совпадала с вектором Вороного и сохранялась правая система координат (надо, чтобы обход области по границе был направлен)
    quadrangles_boundary = quadrangles[boundary]
    quadrangles_boundary_coords = nodes[quadrangles_boundary]

    is_first_V_middle = np.all(np.isclose(quadrangles_boundary_coords[:, ::2].sum(axis=1) / 2, quadrangles_boundary_coords[:, 1]), axis=1)

    quadrangles_boundary_changed = np.concatenate((quadrangles_boundary[:, 2:], quadrangles_boundary[:, :2]), axis=1)
    quadrangles_boundary = np.where(is_first_V_middle.reshape(-1, 1), quadrangles_boundary_changed, quadrangles_boundary)

    quadrangles[boundary] = quadrangles_boundary


def compute_errors(y, u, areas, groups):
    error = y - u

    indexes_D = np.r_[:groups[1], groups[2]:groups[3]]
    indexes_V = np.r_[groups[1]:groups[2]]
    error_D = error[indexes_D]
    error_V = error[indexes_V]

    L2_D = np.sqrt((error_D ** 2 * areas[indexes_D]).sum())
    L2_V = np.sqrt((error_V ** 2 * areas[indexes_V]).sum())
    L2 = np.sqrt(L2_D ** 2 + L2_V ** 2)

    Lmax_D = np.abs(error_D).max()
    Lmax_V = np.abs(error_V).max()
    Lmax = max(Lmax_D, Lmax_V)

    return L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax


def compute_vector_error(g, q, quadrangle_areas):
    vector_error = g - q
    L2_q = np.sum(vector_error ** 2, axis=1) * quadrangle_areas
    L2_q = np.sqrt(L2_q.sum())
    return L2_q


def assemble_ru(r, cell_areas):
    row = np.arange(cell_areas.size)
    data = r * cell_areas
    return row, row, data


# убрать
def assemble_du_dt(tau, yn, cell_areas):
    row = np.arange(cell_areas.size)
    data = 1 / tau * cell_areas
    b = np.zeros(cell_areas.size)
    b = data * yn
    return row, row, data, b

# if matrix_g.shape[-1] == 4:
#     g_local = np.sum(matrix_g * y[quadrangles][:, np.newaxis], axis=2)
def compute_g_local(matrix_g, boundary_vector_g, y, quadrangles, boundary):
    g_local = np.sum(matrix_g * np.pad(y, (0, quadrangles.max() + 1 - y.size))[quadrangles][:, np.newaxis], axis=2)
    g_local[boundary] -= boundary_vector_g
    return g_local


def compute_global_g(B, matrix_g, boundary_vector_g, y, quadrangles, boundary):
    g_local = compute_g_local(matrix_g, boundary_vector_g, y, quadrangles, boundary)
    g = np.squeeze(B @ g_local[:, :, np.newaxis])
    return g


#
def compute_basis_vectors_and_lenghts(quadrangles, nodes):
    quadangle_coords = nodes[quadrangles]
    eD = quadangle_coords[:, 2] - quadangle_coords[:, 0]
    eV = quadangle_coords[:, 3] - quadangle_coords[:, 1]
    lD = np.linalg.norm(eD, axis=1)
    lV = np.linalg.norm(eV, axis=1)
    eD /= lD[:, np.newaxis]
    eV /= lV[:, np.newaxis]
    return eD, eV, lD, lV


def compute_quadrangle_areas(d1, d2):
    return d1 * d2 / 2


def assemble_change_of_basis_matrices(eD, eV):
    return np.stack((eD, eV), axis=-1)


def compute_local_k(global_k, B):
    return np.linalg.inv(B) @ global_k @ B


def assemble_matrix_g(k, lD, lV):
    matrix_g = k.copy()
    matrix_g[:, :, 0] /= lD[:, np.newaxis]
    matrix_g[:, :, 1] /= lV[:, np.newaxis]
    matrix_g = np.tile(matrix_g, 2)
    matrix_g[:, :, 2:] *= -1
    return matrix_g


# def assemble_boundary_matrix_g(k, lD, lV, c3, bc3):
#     kDD = k[:, 0, 0]
#     kDV = k[:, 0, 1]
#     kVD = k[:, 1, 0]
#     kVV = k[:, 1, 1]
    
#     gamma = c3 * kVV / (kVV - c3*lV)

#     boundary_matrix_g = np.zeros((k.shape[0], 2, 5))

#     boundary_matrix_g[:, 1, 0] -= gamma * lV * kVD / (lD * kVV)
#     boundary_matrix_g[:, 1, 1] -= gamma
#     boundary_matrix_g[:, 1, 2] -= boundary_matrix_g[:, 1, 0]
#     boundary_matrix_g[:, 1, 4] = gamma * bc3 / c3

#     boundary_matrix_g[:, 0, 0] -= (kDV*kVD/kVV - kDD) / lD
#     boundary_matrix_g[:, 0, 2] -= boundary_matrix_g[:, 0, 0]
#     boundary_matrix_g[:, 0] += (kDV / kVV)[:, np.newaxis] * boundary_matrix_g[:, 1]

#     return boundary_matrix_g


# def assemble_boundary_matrix_g(k, lD, lV, c3):
#     kDD = k[:, 0, 0]
#     kDV = k[:, 0, 1]
#     kVD = k[:, 1, 0]
#     kVV = k[:, 1, 1]
    
#     gamma = c3 * kVV / (kVV - c3*lV)

#     boundary_matrix_g = np.zeros((k.shape[0], 2, 4))

#     boundary_matrix_g[:, 1, 0] -= gamma * lV * kVD / (lD * kVV)
#     boundary_matrix_g[:, 1, 1] -= gamma
#     boundary_matrix_g[:, 1, 2] -= boundary_matrix_g[:, 1, 0]

#     boundary_matrix_g[:, 0, 0] -= (kDV*kVD/kVV - kDD) / lD
#     boundary_matrix_g[:, 0, 2] -= boundary_matrix_g[:, 0, 0]
#     boundary_matrix_g[:, 0] += (kDV / kVV)[:, np.newaxis] * boundary_matrix_g[:, 1]

#     return boundary_matrix_g


# def assemble_boundary_vector_g(k, lV, c3, bc3):
#     kDV = k[:, 0, 1]
#     kVV = k[:, 1, 1]

#     boundary_vector_g = np.zeros((k.shape[0], 2))
#     gamma = c3 * kVV / (kVV - c3*lV)
#     boundary_vector_g[:, 1] -= gamma * bc3 / c3
#     boundary_vector_g[:, 0] = (kDV / kVV) * boundary_vector_g[:, 1]

#     return boundary_vector_g


def assemble_boundary_matrix_g(k, lD, lV, c3):
    kDD = k[:, 0, 0]
    kDV = k[:, 0, 1]
    kVD = k[:, 1, 0]
    kVV = k[:, 1, 1]
    
    gamma = c3 / (kVV + c3*lV)

    boundary_matrix_g = np.zeros((k.shape[0], 2, 4))

    boundary_matrix_g[:, 0, 2] = 1 / lD
    boundary_matrix_g[:, 0, 0] -= boundary_matrix_g[:, 0, 2]

    boundary_matrix_g[:, 1, 0::2] -= (gamma * lV*kVD)[:, np.newaxis] * boundary_matrix_g[:, 0, 0::2]
    boundary_matrix_g[:, 1, 1] = kVV * gamma

    boundary_matrix_g[:, 0, 0::2] *= (kDV*kVD/kVV - kDD)[:, np.newaxis]
    boundary_matrix_g[:, 0] += (kDV / kVV)[:, np.newaxis] * boundary_matrix_g[:, 1]

    return boundary_matrix_g


def assemble_boundary_vector_g(k, lV, c3, bc3):
    kDV = k[:, 0, 1]
    kVV = k[:, 1, 1]

    boundary_vector_g = np.zeros((k.shape[0], 2))
    boundary_vector_g[:, 1] = kVV * bc3 / (kVV + c3*lV)
    boundary_vector_g[:, 0] = (kDV / kVV) * boundary_vector_g[:, 1]

    return boundary_vector_g


def join_matrix_g(inner_matrix_g, boundary_matrix_g, inner, boundary):
    matrix_g = np.zeros((inner_matrix_g.shape[0] + boundary_matrix_g.shape[0], 2, 4))
    matrix_g[inner] = inner_matrix_g
    matrix_g[boundary] = boundary_matrix_g
    return matrix_g


def assemble_integral_matrix(matrix_g, lD, lV):
    integral_matrix = matrix_g.copy()
    integral_matrix[:, 0] *= lV[:, np.newaxis]
    integral_matrix[:, 1] *= lD[:, np.newaxis]
    integral_matrix = np.tile(integral_matrix, (1, 2, 1))
    integral_matrix[:, 2:] *= -1
    return integral_matrix


def assemble_boundary_integral_vector(boundary_vector_g, lD, lV):
    boundary_integral_vector = boundary_vector_g.copy()
    boundary_integral_vector[:, 0] *= lV
    boundary_integral_vector[:, 1] *= lD
    boundary_integral_vector = np.tile(boundary_integral_vector, 2)
    boundary_integral_vector[:, 2:] *= -1
    return boundary_integral_vector


# def add_boundary_integrals_to_matrix(integral_matrix, lD, boundary, c0, c2, bc0, bc2):
#     """Интегралы по границе области"""
#     half_lD = lD[boundary] / 2
#     integral_matrix[boundary, 0, 0] += half_lD * -c0
#     integral_matrix[boundary, 0, 4] += half_lD * bc0
#     integral_matrix[boundary, 2, 2] += half_lD * -c2
#     integral_matrix[boundary, 2, 4] += half_lD * bc2
#     return integral_matrix


# def add_boundary_integrals_to_matrix(integral_matrix, lD, boundary, c0, c2):
#     half_lD = lD[boundary] / 2
#     integral_matrix[boundary, 0, 0] -= half_lD * c0
#     integral_matrix[boundary, 2, 2] -= half_lD * c2
#     return integral_matrix


# def add_boundary_integrals_to_vector(boundary_integral_vector, lD, bc0, bc2):
#     half_lD = lD / 2
#     boundary_integral_vector[:, 0] -= half_lD * bc0
#     boundary_integral_vector[:, 2] -= half_lD * bc2
#     return boundary_integral_vector

# def add_boundary_integrals_to_matrix(integral_matrix, lD, boundary, c0, c2, c3):
#     half_lD = lD[boundary] / 2
#     integral_matrix[boundary, 0, 0] += half_lD * c0
#     integral_matrix[boundary, 2, 2] += half_lD * c2
#     return integral_matrix


# def add_boundary_integrals_to_vector(boundary_integral_vector, lD, bc0, bc2, bc3):
#     half_lD = lD / 2
#     boundary_integral_vector[:, 0] += half_lD * bc0
#     boundary_integral_vector[:, 2] += half_lD * bc2
#     return boundary_integral_vector


def add_boundary_integrals_to_matrix(integral_matrix, lD, boundary, c3):
    half_lD = lD[boundary] / 2
    integral_matrix[boundary, 0::2, 0::2] += (c3/2 * half_lD)[:, np.newaxis, np.newaxis]
    return integral_matrix


def add_boundary_integrals_to_vector(boundary_integral_vector, lD, bc3):
    half_lD = lD / 2
    boundary_integral_vector[:, 0] += bc3 * half_lD
    boundary_integral_vector[:, 2] += bc3 * half_lD
    return boundary_integral_vector


def assemble_coo_sparse_format(quadrangles, integral_matrix):
    row = np.repeat(quadrangles, 4)
    col = np.tile(quadrangles, 4).flatten()
    data = integral_matrix.flatten()
    return row, col, data


# def assemble_vector_b(integral_matrix, quadrangles, boundary, full_size):
#     b = np.zeros(full_size)
#     b[quadrangles[boundary, 0]] -= integral_matrix[boundary, 0, 4]
#     b[quadrangles[boundary, 1]] -= integral_matrix[boundary, 1, 4]
#     b[quadrangles[boundary, 2]] -= integral_matrix[boundary, 2, 4]
#     return b


def assemble_vector_b(boundary_integral_vector, quadrangles, boundary, full_size):
    b = np.zeros(full_size)
    for i in range(3):
        b[quadrangles[boundary, i]] += boundary_integral_vector[:, i]
    return b
