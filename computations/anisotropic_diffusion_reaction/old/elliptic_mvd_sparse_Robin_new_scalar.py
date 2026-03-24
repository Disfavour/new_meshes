import gmsh
import numpy as np
import sympy
import sys
#sys.path.append('mesh_generation')
#import utility
import utility_mvd
from scipy.sparse import lil_array, coo_array, csr_array
from scipy.sparse.linalg import spsolve
import time
import pathlib


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
    areas_V = utility_mvd.compute_polygon_area_vectorized(nodes[cells_V])

    cells_D = cells[np.r_[:groups[1], groups[2]:groups[3]]]
    cell_sizes = np.fromiter((cell.size for cell in cells_D), dtype=int, count=cells_D.shape[0])
    areas_D = np.empty(groups[1] + groups[3] - groups[2])
    for number_of_nodes in np.unique(cell_sizes):
        indexes = cell_sizes == number_of_nodes
        cells_specified = cells_D[indexes].astype((int, (number_of_nodes,)), copy=False)
        areas_specified = utility_mvd.compute_polygon_area_vectorized(nodes[cells_specified])
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


def assemble_div_q(k, quadrangles, quadangle_coords, quadrangle_centers, inner, boundary, bc, c, nodes, groups):
    eD = quadangle_coords[:, 2] - quadangle_coords[:, 0]
    eV = quadangle_coords[:, 3] - quadangle_coords[:, 1]
    lD = np.linalg.norm(eD, axis=1)
    lV = np.linalg.norm(eV, axis=1)
    eD /= lD.reshape(-1, 1)
    eV /= lV.reshape(-1, 1)

    B = np.stack((eD, eV), axis=-1)

    #bc1 = bc(*quadrangle_centers[boundary].T, *eV[boundary].T)
    k = k(*quadrangle_centers.T)
    if np.isscalar(k):
        k = np.full(quadrangle_centers.shape[0], k)
    c_centers = c(*quadrangle_centers[boundary].T)

    g_matrix_inner = np.zeros((quadrangles[inner].shape[0], 2, 4))
    g_matrix_inner[:, 0, 0] = k[inner] / lD[inner]
    g_matrix_inner[:, 0, 2] -= g_matrix_inner[:, 0, 0]
    g_matrix_inner[:, 1, 1] = k[inner] / lV[inner]
    g_matrix_inner[:, 1, 3] -= g_matrix_inner[:, 1, 1]

    integral_matrix_inner = g_matrix_inner.copy()
    integral_matrix_inner[:, 0] *= lV[inner].reshape(-1, 1)
    integral_matrix_inner[:, 1] *= lD[inner].reshape(-1, 1)
    integral_matrix_inner = np.tile(integral_matrix_inner, (1, 2, 1))
    integral_matrix_inner[:, 2:] *= -1

    row_inner = np.repeat(quadrangles[inner], 4)
    col_inner = np.tile(quadrangles[inner], 4).reshape(-1)
    data_inner = integral_matrix_inner.reshape(-1)

    g_matrix_boundary = np.zeros((quadrangles[boundary].shape[0], 2, 4))
    g_matrix_boundary[:, 0, 0] = k[boundary] / lD[boundary]
    g_matrix_boundary[:, 0, 2] -= g_matrix_boundary[:, 0, 0]
    alpha = k[boundary] / (k[boundary] + c_centers*lV[boundary])
    g_matrix_boundary[:, 1, 1] = alpha * c_centers
    g_matrix_boundary[:, 1, 3] = alpha * bc(*quadrangle_centers[boundary].T, *eV[boundary].T)

    integral_matrix_boundary = g_matrix_boundary.copy()
    integral_matrix_boundary[:, 0] *= lV[boundary].reshape(-1, 1)
    integral_matrix_boundary[:, 1] *= lD[boundary].reshape(-1, 1)
    integral_matrix_boundary = np.concatenate((integral_matrix_boundary, -integral_matrix_boundary[:, 0].reshape(-1, 1, 4)), axis=1)

    half_lD = lD[boundary] / 2
    integral_matrix_boundary[:, 0, 0] += c(*nodes[quadrangles[boundary, 0]].T) * half_lD
    integral_matrix_boundary[:, 0, 3] = bc(*nodes[quadrangles[boundary, 0]].T, *eV[boundary].T) * half_lD
    integral_matrix_boundary[:, 2, 2] += c(*nodes[quadrangles[boundary, 2]].T) * half_lD
    integral_matrix_boundary[:, 2, 3] = bc(*nodes[quadrangles[boundary, 2]].T, *eV[boundary].T) * half_lD

    row_boundary = np.repeat(quadrangles[boundary, :3], 3)
    col_boundary = np.tile(quadrangles[boundary, :3], 3).reshape(-1)
    data_boundary = integral_matrix_boundary[:, :, :3].reshape(-1)

    row = np.concatenate((row_inner, row_boundary))
    col = np.concatenate((col_inner, col_boundary))
    data = np.concatenate((data_inner, data_boundary))

    b = np.zeros(groups[3])
    b[quadrangles[boundary, 0]] -= integral_matrix_boundary[:, 0, 3]
    b[quadrangles[boundary, 1]] -= integral_matrix_boundary[:, 1, 3]
    b[quadrangles[boundary, 2]] -= integral_matrix_boundary[:, 2, 3]

    # b[quadrangles[boundary, :3]] -= integral_matrix_boundary[:, :, 3] не работает правильно

    return row, col, data, b, g_matrix_inner, g_matrix_boundary, B


def assemble_ru(r, areas, nodes):
    row = np.arange(nodes.shape[0])
    data = r(*nodes.T) * areas
    return row, row, data


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


def compute_vector_errors(g, q, quadrangle_areas):
    vector_error = g - q
    L2_q = np.sum(vector_error ** 2, axis=1) * quadrangle_areas
    L2_q = np.sqrt(L2_q.sum())
    return L2_q


def solve_problem(k, r, f, c, bc, u, q, quadrangle_mesh_fname):
    t = [time.time()]

    nodes, quadrangles = load_msh(quadrangle_mesh_fname)
    #t.append(time.time())

    groups, cells = load_npz(pathlib.Path(quadrangle_mesh_fname).with_suffix('.npz'))
    t.append(time.time())

    u = u(*nodes[:groups[3]].T)

    quadrangle_areas = utility_mvd.compute_polygon_area_vectorized(nodes[quadrangles])

    cell_areas = compute_cell_areas(cells, groups, nodes)

    boundary = (quadrangles[:, 0] >= groups[2]) & (quadrangles[:, 2] >= groups[2])
    inner = ~boundary

    redirect_eV_on_boundary(quadrangles, nodes, boundary)

    quadrangle_centers = utility_mvd.compute_intersection_points(nodes[quadrangles])
    row, col, data, b_div, g_matrix_inner, g_matrix_boundary, B = assemble_div_q(k, quadrangles, nodes[quadrangles], quadrangle_centers, inner, boundary, bc, c, nodes, groups)
    
    row_r, col_r, data_r = assemble_ru(r, cell_areas, nodes[:groups[3]])

    row = np.concatenate((row, row_r))
    col = np.concatenate((col, col_r))
    data = np.concatenate((data, data_r))

    A = coo_array((data, (row, col)), shape=(groups[3], groups[3]))
    A.eliminate_zeros()
    A = A.tocsr()

    # tmp = A - A.T
    # print(f'Symmetry {tmp.min()} {tmp.max()}')

    # if A.shape[0] < 1e5:
    #     tmp = A.todense()
    #     check1 = np.allclose(tmp[:groups[1], groups[1]:groups[2]], 0)
    #     check2 = np.allclose(tmp[groups[2]:groups[3], groups[1]:groups[2]], 0)
    #     check3 = np.allclose(tmp[groups[1]:groups[2], :groups[1]], 0)
    #     check4 = np.allclose(tmp[groups[1]:groups[2], groups[2]:groups[3]], 0)

    #     check = (check1, check2, check3, check4)

    #     print(f'Check separation D&V {np.all(check)} {check}')

    # np.savetxt('A_scalar_new.txt', A.toarray(), fmt='%5.2f')
    # np.savetxt('A1_Neumann_dif.txt', tmp.toarray(), fmt='%5.2f')
    # print(groups)

    b = f(*nodes[:groups[3]].T) * cell_areas + b_div

    t.append(time.time())
    y = spsolve(A, b)
    t.append(time.time())

    L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = compute_errors(y, u, cell_areas, groups)

    
    q = q(quadrangle_centers[:, 0], quadrangle_centers[:, 1]).T.reshape(-1, 2)
    g_inner = np.sum(g_matrix_inner * y[quadrangles[inner]].reshape(-1, 1, 4), axis=2)
    g_boundary = np.sum(g_matrix_boundary[:, :, :3] * y[quadrangles[boundary, :3]].reshape(-1, 1, 3), axis=2) + g_matrix_boundary[:, :, 3]

    g = np.zeros((quadrangles.shape[0], 2))
    g[inner] = g_inner
    g[boundary] = g_boundary

    g = np.reshape(B @ g.reshape(-1, 2, 1), (-1, 2))

    L2_q = compute_vector_errors(g, q, quadrangle_areas)
    
    t.append(time.time())
    time_load = t[1] - t[0]
    time_solve = t[3] - t[2]
    time_total = t[-1] - t[0]
    time_other = time_total - time_load - time_solve
    print(f'Time\ttotal {time_total:6.2f}')
    print(f'\tload  {time_load:6.2f} ({time_load / time_total:6.2%})')
    print(f'\tsolve {time_solve:6.2f} ({time_solve / time_total:6.2%})')
    print(f'\tother {time_other:6.2f} ({time_other / time_total:6.2%})')

    return nodes.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q


def setup_problem():
    # div (-k * grad u) + r*u = f
    # q*n - cu = bc
    x, y, nx, ny = sympy.symbols('x y nx ny')
    n = sympy.Matrix([nx, ny])

    u = sympy.exp(x*y)
    k = 1
    r = 1
    c = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) - c*u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, c, bc, u, q


if __name__ == '__main__':
    res = solve_problem(*setup_problem(), 'meshes/rectangle/rectangle_8_quadrangle.msh')
    print(*res)

    # ошибка не падает с увеличением сетки, где-то ошибся
    # теперь ошибка по делоне нормас, но с вороным чета

    # при с -> 0 матрица становится боле симметричной а при увеличении с наоборот. Можем мы стримимся в дирихле, а там нужен лифтинг?

    # Если к симметричная, то матрица симметричная, а если нет, то нет
    # В дирихле при любой к симметричная (после лифтинга)

    # 193801 1.9562637730867953e-07 1.7184392399539124e-07 2.603843576581582e-07 2.0100485735863316e-06 2.3723977249368033e-06 2.3723977249368033e-06 2.5126142249882246e-05
    # 193801 6.146895727608258e-06  9.119710596786576e-06  1.099788381713637e-05 4.059213710538856e-05  1.439699248040327e-05  4.059213710538856e-05  3.115950781353237e-05


    # -6.470747271336563e-10 6.470747271336563e-10
    # Time    total 497.28
    #         load  11.09 (2.23%)
    #         solve 478.39 (96.20%)
    #         other 7.80 (1.57%)
    # 3265828 6.331006340791239e-07 9.953358150231743e-08 6.408770137226542e-07 5.919549765920351e-06 1.9154763553075327e-07 5.919549765920351e-06 3.3768098206533204e-06