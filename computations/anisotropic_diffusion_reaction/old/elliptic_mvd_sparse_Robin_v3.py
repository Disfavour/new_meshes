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


def load_data(quadrangle_mesh_name):
    gmsh.initialize()
    gmsh.open(f'{quadrangle_mesh_name}.msh')

    quads, quad_nodes = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    quad_nodes = quad_nodes.reshape(-1, 4)

    nodes, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)

    quad_areas = gmsh.model.mesh.get_element_qualities(quads, 'volume')

    gmsh.finalize()

    assert nodes.size == nodes.max()
    if not np.all(nodes[:-1] < nodes[1:]):
        indices = np.argsort(nodes)
        nodes = nodes[indices]
        node_coords = node_coords[indices]
    assert np.all(nodes[:-1] < nodes[1:])
    
    loaded = np.load(f'{quadrangle_mesh_name}.npz', allow_pickle=True)
    node_groups = loaded['node_groups']
    cell_nodes = loaded['cells']

    node_coords = node_coords[:, :2]
    quad_nodes = quad_nodes.astype(int) - 1
    node_groups = node_groups.astype(int)
    cell_nodes = np.array([nodes.astype(int) for nodes in cell_nodes], dtype=object) - 1

    return node_coords, quad_nodes, node_groups, cell_nodes, quad_areas


def create_local_coordinate_systems(k, quadrangles, coords, inner, boundary, bc, c):
    quadangles_coords = coords[quadrangles]

    e_Ds = quadangles_coords[:, 2] - quadangles_coords[:, 0]
    e_Vs = quadangles_coords[:, 3] - quadangles_coords[:, 1]

    e_D_lenghts = np.linalg.norm(e_Ds, axis=1).reshape(-1, 1)
    e_V_lenghts = np.linalg.norm(e_Vs, axis=1).reshape(-1, 1)
    
    e_Ds /= e_D_lenghts
    e_Vs /= e_V_lenghts

    Bs = np.stack((e_Ds, e_Vs), axis=-1)

    ks = np.linalg.inv(Bs) @ k @ Bs

    q_coeffs_inner = ks[inner]
    q_coeffs_inner[:, :, 0] /= e_D_lenghts[inner]
    q_coeffs_inner[:, :, 1] /= e_V_lenghts[inner]
    q_coeffs_inner = np.tile(q_coeffs_inner, 2)
    q_coeffs_inner[:, :, :2] *= -1

    integral_coeffs = q_coeffs_inner.copy()
    integral_coeffs[:, 0] *= e_V_lenghts[inner]
    integral_coeffs[:, 1] *= e_D_lenghts[inner]
    integral_coeffs = np.tile(integral_coeffs, (1, 2, 1))
    integral_coeffs[:, 2:] *= -1
    integral_coeffs *= -1

    row_inner = np.repeat(quadrangles[inner], 4)
    col_inner = np.tile(quadrangles[inner], 4)
    data_inner = integral_coeffs

    
    kb = ks[boundary]

    lD = e_D_lenghts[boundary].reshape(-1)
    lV = e_V_lenghts[boundary].reshape(-1)

    kDD = kb[:, 0, 0]
    kDV = kb[:, 0, 1]
    kVD = kb[:, 1, 0]
    kVV = kb[:, 1, 1]

    bc3 = bc(*coords[quadrangles[boundary, 3]].T, *e_Vs[boundary].T)

    q_coeffs_boundary = np.zeros((quadrangles[boundary].shape[0], 2, 3))
    q_coeffs_boundary[:, 0, 0] = -kDD/lD + kDV*kVD/(lD*kVV)
    q_coeffs_boundary[:, 0, 1] = -c*kDV/kVV
    q_coeffs_boundary[:, 0, 2] = -q_coeffs_boundary[:, 0, 0]
    q_coeffs_boundary[:, 1, 1] = -c

    q_b = np.zeros((quadrangles[boundary].shape[0], 2))
    q_b[:, 0] = bc3*kDV/kVV
    q_b[:, 1] = bc3

    # тут на пол ребра делоне будем использовать граничное в этом узле Делоне, а не в вороном
    integral_coeffs_boundary = np.zeros((quadrangles[boundary].shape[0], 3, 3))
    integral_coeffs_boundary[:, 0] = q_coeffs_boundary[:, 0] * e_V_lenghts[boundary]
    integral_coeffs_boundary[:, 1] = q_coeffs_boundary[:, 1] * e_D_lenghts[boundary]
    integral_coeffs_boundary[:, 2] = -integral_coeffs_boundary[:, 0]
    #integral_coeffs_boundary[:, 0::2] += (integral_coeffs_boundary[:, 1] / 2).reshape(-1, 1, 3)

    # Сделали симметрично, но правую часть как подправлять?
    # integral_coeffs_boundary[:, 1, 0] = integral_coeffs_boundary[:, 0, 1]
    # integral_coeffs_boundary[:, 1, 2] = integral_coeffs_boundary[:, 2, 1]

    bc0 = bc(*coords[quadrangles[boundary, 0]].T, *e_Vs[boundary].T)
    bc2 = bc(*coords[quadrangles[boundary, 2]].T, *e_Vs[boundary].T)
    half = (e_D_lenghts[boundary] / 2).reshape(-1)
    integral_coeffs_boundary[:, 0, 0] += -c*half
    integral_coeffs_boundary[:, 2, 2] += -c*half

    integral_b = np.zeros((quadrangles[boundary].shape[0], 3))
    integral_b[:, 0] = q_b[:, 0] * lV
    integral_b[:, 1] = q_b[:, 1] * lD
    integral_b[:, 2] = -integral_b[:, 0]
    #integral_b[:, 0::2] += (integral_b[:, 1] / 2).reshape(-1, 1)

    integral_b[:, 0] += bc0*half
    integral_b[:, 2] += bc2*half

    # # boundary of area Omega
    # half = (e_D_lenghts[boundary] / 2).reshape(-1)
    # integral_kefs = np.zeros((quadrangles[boundary].shape[0], 2))
    # integral_kefs[:] = (-c * half).reshape(-1, 1)

    # integral_b = np.zeros((quadrangles[boundary].shape[0], 2))
    # integral_b[:, 0] = bc0 * half
    # integral_b[:, 1] = bc2 * half
    
    integral_coeffs_boundary *= -1
    integral_b *= -1

    b = np.zeros(coords.shape[0])
    b[quadrangles[boundary, 0]] += integral_b[:, 0]
    b[quadrangles[boundary, 1]] += integral_b[:, 1]
    b[quadrangles[boundary, 2]] += integral_b[:, 2]

    row_boundary = np.repeat(quadrangles[boundary, :3], 3)
    col_boundary = np.tile(quadrangles[boundary, :3], 3)
    data_boundary = integral_coeffs_boundary

    row = np.concatenate((row_inner, row_boundary))
    col = np.concatenate((col_inner.flat, col_boundary.flat))
    data = np.concatenate((data_inner.flat, data_boundary.flat))

    return row, col, data, b, Bs, q_coeffs_inner, q_coeffs_boundary, q_b


def approximate_r_u(r, areas, coords):
    nodes = np.arange(coords.shape[0])
    data = r * areas
    return nodes, nodes, data


def compute_errors(u, ue, areas, groups):
    error = u - ue

    indexes_D = np.r_[:groups[0], groups[1]:groups[2]]
    indexes_V = np.r_[groups[0]:groups[1]]
    error_D = error[indexes_D]
    error_V = error[indexes_V]

    L2_D = np.sqrt((error_D ** 2 * areas[indexes_D]).sum())
    L2_V = np.sqrt((error_V ** 2 * areas[indexes_V]).sum())
    L2 = np.sqrt(L2_D ** 2 + L2_V ** 2)

    Lmax_D = np.abs(error_D).max()
    Lmax_V = np.abs(error_V).max()
    Lmax = max(Lmax_D, Lmax_V)

    return L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax


def compute_vector_errors(q, qe, quadrangle_areas):
    vector_error = q - qe
    L2_q = np.sum(vector_error ** 2, axis=1) * quadrangle_areas
    L2_q = np.sqrt(L2_q.sum())
    return L2_q


def calculate(quadrangle_mesh_name, k, c, info=False, plot=False):
    x, y, nx, ny = sympy.symbols('x y nx ny')
    u = sympy.exp(x*y) # x * (x - 1) * sympy.sin(4 * sympy.pi * y / 3)
    r = 1
    #c = 1

    # - div (k * grad u) + r*u = f
    n = sympy.Matrix([nx, ny])
    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], -div + r * u, "numpy")
    u_exact = sympy.lambdify([x, y], u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n) + c*u, "numpy")
    q_exact = sympy.lambdify([x, y], q, "numpy")
    
    t = [time.time()]
    coords, quadrangles, groups, cells, quadrangle_areas = load_data(quadrangle_mesh_name)
    t.append(time.time())

    ue = u_exact(*coords[:groups[2]].T)

    # areas
    areas = np.empty(groups[2])

    cells_V = cells[groups[0]:groups[1]].astype((int, (3,)), copy=False)
    areas_V = utility_mvd.compute_polygon_area_vectorized(coords[cells_V])

    cells_D = cells[np.r_[:groups[0], groups[1]:groups[2]]]
    cell_sizes = np.fromiter((cell.size for cell in cells_D), dtype=int, count=cells_D.shape[0])
    areas_D = np.empty(groups[0] + groups[2] - groups[1])
    for number_of_nodes in np.unique(cell_sizes):
        indexes = cell_sizes == number_of_nodes
        cells_specified = cells_D[indexes].astype((int, (number_of_nodes,)), copy=False)
        areas_specified = utility_mvd.compute_polygon_area_vectorized(coords[cells_specified])
        areas_D[indexes] = areas_specified

    areas = np.concatenate((areas_D[:groups[0]], areas_V, areas_D[groups[0]:]))

    # Будем через матрицы кефов делать

    # quads boundary - c 2 делоне вершинами на границе
    boundary = (quadrangles[:, 0] >= groups[1]) & (quadrangles[:, 2] >= groups[1])
    inner = ~boundary

    # сделаем чтобы e_V указывал внешнюю нормаль 
    quadrangles_boundary = quadrangles[boundary]
    quadrangles_boundary_coords = coords[quadrangles_boundary]

    is_first_V_middle = np.all(np.isclose(quadrangles_boundary_coords[:, ::2].sum(axis=1) / 2, quadrangles_boundary_coords[:, 1]), axis=1)

    n_nodes = np.where(is_first_V_middle.reshape(-1, 1), quadrangles_boundary[:, 3::-2], quadrangles_boundary[:, 1::2])

    n_node_coords = coords[n_nodes]
    n = n_node_coords[:, 1] - n_node_coords[:, 0]
    # n /= np.linalg.norm(n, axis=1).reshape(-1, 1)

    # мы поменяли только вектор e_v, тогда нарушиться правая система координат, но пофиг по идее)
    quadrangles[boundary, 1::2] = n_nodes

    # Сделаем чтобы e_D, e_V снова были по часовой

    # Матрица поворота по часовой на 90 градусов
    M = np.array(( 
        (0, 1),
        (-1, 0)
    ))
    
    e_Ds = quadrangles_boundary_coords[:, 2] - quadrangles_boundary_coords[:, 0]

    n = M @ n.T
    n = n.T

    e_D_nodes = np.where((np.sum(n * e_Ds, axis=1) > 0).reshape(-1, 1), quadrangles_boundary[:, ::2], quadrangles_boundary[:, 2::-2])
    quadrangles[boundary, ::2] = e_D_nodes

    row, col, data, b_div, Bs, q_coeffs_inner, q_coeffs_boundary, q_b = create_local_coordinate_systems(k, quadrangles, coords, inner, boundary, bc, c)

    #row, col, data = approximate_div_q(integral_coeffs, quadrangles, quadrangle_mask_inner, quadrangle_mask_boundary, coords, q_exact)
    row_r, col_r, data_r = approximate_r_u(r, areas, coords[:groups[2]])

    row = np.concatenate((row, row_r))
    col = np.concatenate((col, col_r))
    data = np.concatenate((data, data_r))

    A = coo_array((data, (row, col)), shape=(groups[2], groups[2]))
    A.eliminate_zeros()
    A = A.tocsr()

    # check symmetry
    tmp = A - A.T
    print(np.abs(tmp).max(), tmp.min(), tmp.max())

    # np.savetxt('A2_Neumann.txt', A.toarray(), fmt='%5.2f')
    # np.savetxt('A2_Neumann_dif.txt', tmp.toarray(), fmt='%5.2f')
    # print(groups)

    b = f(coords[:groups[2], 0], coords[:groups[2], 1]) * areas[:groups[2]] - b_div[:groups[2]]

    t.append(time.time())
    u = spsolve(A, b)
    t.append(time.time())

    L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = compute_errors(u, ue, areas, groups)

    quadrangle_centers = utility_mvd.compute_intersection_points(coords[quadrangles])
    q_e = q_exact(quadrangle_centers[:, 0], quadrangle_centers[:, 1]).T.reshape(-1, 2)
    #print(q_b.shape)
    q_local_inner = np.sum(q_coeffs_inner * u[quadrangles[inner]].reshape(-1, 1, 4), axis=2)
    q_local_boundary = np.sum(q_coeffs_boundary * u[quadrangles[boundary, :3]].reshape(-1, 1, 3), axis=2) + q_b

    #print(q_coeffs_boundary.shape, u[quadrangles[boundary, ::2]].shape, q_local_boundary.shape, q_b.shape)
    #exit()

    q_local = np.zeros((quadrangles.shape[0], 2))
    q_local[inner] = q_local_inner
    q_local[boundary] = q_local_boundary

    #q_local = np.sum(q_coeffs * u[quadrangles].reshape(-1, 1, 4), axis=2)

    #print(q_local.shape, Bs.shape)
    #print((Bs @ q_local.reshape(-1, 2, 1)).shape)
    q = np.reshape(Bs @ q_local.reshape(-1, 2, 1), (-1, 2))

    # print(q_e[boundary, 1])
    # print(q[boundary, 1])

    # print((q_coeffs_boundary * u[quadrangles[boundary, ::2]].reshape(-1, 1, 2)).reshape(-1, 2))
    # print(q_b)
    # print(q_local_boundary)

    L2_q = compute_vector_errors(q, q_e, quadrangle_areas)
    
    t.append(time.time())
    time_load = t[1] - t[0]
    time_solve = t[3] - t[2]
    time_total = t[-1] - t[0]
    time_other = time_total - time_load - time_solve
    print(f'Time\ttotal {time_total:6.2f}')
    print(f'\tload  {time_load:6.2f} ({time_load / time_total:6.2%})')
    print(f'\tsolve {time_solve:6.2f} ({time_solve / time_total:6.2%})')
    print(f'\tother {time_other:6.2f} ({time_other / time_total:6.2%})')

    if plot:
        utility_mvd.plot_results(u, ue, coords[:groups[2]], groups)

    return coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q


if __name__ == '__main__':
    k = np.array(
        ((1, 0.0),
         (0.0, 2))
    )
    res = calculate('meshes/rectangle/rectangle_1_quadrangle', k, 1000, info=True, plot=False)
    print(*res)

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