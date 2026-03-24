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


# # попробовать оптимизировать (повыносить из цикла, векторизовать через нампай)
# def approximate_div_k_grad(row, col, data, quadrangles, coords, k_global):
#     for quadrangle in quadrangles:
#         e_D = coords[quadrangle[2]] - coords[quadrangle[0]]
#         e_D_lenght = np.linalg.norm(e_D)

#         e_V = coords[quadrangle[3]] - coords[quadrangle[1]]
#         e_V_lenght = np.linalg.norm(e_V)
        
#         B = np.column_stack((
#             e_D / e_D_lenght,
#             e_V / e_V_lenght
#         ))

#         k = np.linalg.inv(B) @ k_global @ B

#         k[0, 0] *= e_V_lenght / e_D_lenght
#         k[1, 1] *= e_D_lenght / e_V_lenght

#         intetgral_col = (quadrangle[2], quadrangle[0], quadrangle[3], quadrangle[1])
#         integral_D_data = -np.array((k[0, 0], -k[0, 0], k[0, 1], -k[0, 1]))
#         integral_V_data = -np.array((k[1, 0], -k[1, 0], k[1, 1], -k[1, 1]))
        
#         row.extend([quadrangle[0]]*4 + [quadrangle[2]]*4 + [quadrangle[1]]*4 + [quadrangle[3]]*4)
#         col.extend(intetgral_col*4)
#         data.extend((*integral_D_data, *-integral_D_data, *integral_V_data, *-integral_V_data))


# row, col, data = [], [], []

# for quadrangle, k in zip(quadrangles, q_integral):
#     intetgral_col = quadrangle
#     integral_D_data = np.array((k[0, 0], k[0, 1], -k[0, 0], -k[0, 1]))
#     integral_V_data = np.array((k[1, 0], k[1, 1], -k[1, 0], -k[1, 1]))
    
#     row.extend(np.repeat(quadrangle, 4))
#     col.extend(np.tile(intetgral_col, 4))
#     data.extend((*integral_D_data, *integral_V_data, *-integral_D_data, *-integral_V_data))

# print(np.allclose(row, row1))
# print(np.allclose(col, col1))
# print(np.allclose(data, data1))


def create_local_coordinate_systems(k, quadrangles, coords, inner, boundary, bc):
    quadangles_coords = coords[quadrangles]

    e_Ds = quadangles_coords[:, 2] - quadangles_coords[:, 0]
    e_Vs = quadangles_coords[:, 3] - quadangles_coords[:, 1]

    e_D_lenghts = np.linalg.norm(e_Ds, axis=1).reshape(-1, 1)
    e_V_lenghts = np.linalg.norm(e_Vs, axis=1).reshape(-1, 1)
    
    e_Ds /= e_D_lenghts
    e_Vs /= e_V_lenghts

    Bs = np.stack((e_Ds, e_Vs), axis=-1)

    ks = np.linalg.inv(Bs) @ k @ Bs

    # tmp = ks.transpose(0, 2, 1)
    # print(np.allclose(ks, tmp))
    
    # for i, j in zip(ks, tmp):
    #     print(i)
    #     print(j)
    #     print()
    # exit()

    q_coeffs_inner = ks[inner]
    q_coeffs_inner[:, :, 0] /= e_D_lenghts[inner]
    q_coeffs_inner[:, :, 1] /= e_V_lenghts[inner]
    q_coeffs_inner = np.tile(q_coeffs_inner, 2)
    q_coeffs_inner[:, :, :2] *= -1

    integral_coeffs = q_coeffs_inner.copy()
    integral_coeffs *= -1
    integral_coeffs[:, 0] *= e_V_lenghts[inner]
    integral_coeffs[:, 1] *= e_D_lenghts[inner]
    integral_coeffs = np.tile(integral_coeffs, (1, 2, 1))
    integral_coeffs[:, 2:] *= -1

    row_inner = np.repeat(quadrangles[inner], 4)
    col_inner = np.tile(quadrangles[inner], 4)
    data_inner = integral_coeffs

    # boundary
    
    ks_b = ks[boundary]
    
    kefs2 = ks_b[:, 0, 1] / ks_b[:, 1, 1]
    kefs = (ks_b[:, 0, 0] - ks_b[:, 0, 1] * ks_b[:, 1, 0] / ks_b[:, 1, 1]) / e_D_lenghts[boundary].reshape(-1)

    # kefs2 = ks_b[:, 0, 1] * e_V_lenghts[boundary].reshape(-1) / (e_D_lenghts[boundary].reshape(-1) * ks_b[:, 1, 1])
    # kefs = (ks_b[:, 0, 0] - ks_b[:, 0, 1] * ks_b[:, 1, 0] / ks_b[:, 1, 1]) * e_V_lenghts[boundary].reshape(-1) / e_D_lenghts[boundary].reshape(-1)

    # print(coords[quadrangles[boundary, 3]].shape)
    # print(q_exact(*coords[quadrangles[boundary, 3]].T).shape)
    # print(e_Vs[boundary].shape)
    #q_V = np.sum(e_Vs[boundary] * q_exact(*coords[quadrangles[boundary, 3]].T).T.reshape(-1, 2), axis=1) #u_boundary
    q_V = bc(*coords[quadrangles[boundary, 3]].T, *e_Vs[boundary].T)

    # print(q_V.shape)
    # print(e_Vs.shape)
    # exit()

    kefs2 *= q_V

    # тока по делоне
    q_coeffs_boundary = np.zeros((quadrangles[boundary].shape[0], 2, 2))
    
    q_coeffs_boundary[:, 0, 0] = -kefs
    q_coeffs_boundary[:, 0, 1] = kefs

    # с минусом, типо справа вектор после равно
    q_b = np.zeros((quadrangles[boundary].shape[0], 2))
    q_b[:, 0] = -kefs2
    q_b[:, 1] = -q_V

    # Вообще хватило бы (quadrangles.shape[0], 3, 3), т.к. 3 нода только
    # в квадранглс по 4 нода, надо сделать чтобы при умножении там нули были, но мы не сможем взять u[quadrangles] ведь в u их тупо не будет
    integral_kefs = np.zeros((quadrangles[boundary].shape[0], 2, 2))
    integral_kefs[:, 0] = q_coeffs_boundary[:, 0] * e_V_lenghts[boundary]
    integral_kefs[:, 1] = -integral_kefs[:, 0]
    integral_kefs *= -1

    integral_b = np.zeros((quadrangles[boundary].shape[0], 3))
    integral_b[:, 0] = q_b[:, 0] * e_V_lenghts[boundary].reshape(-1)
    integral_b[:, 2] = -integral_b[:, 0]
    integral_b[:, 1] = q_b[:, 1] * e_D_lenghts[boundary].reshape(-1)
    integral_b *= -1

    # integral_b не нужен (не нужно получать интегралы в 4-х угольниках), а надо b глобальное
    b = np.zeros(coords.shape[0])
    b[quadrangles[boundary, 0]] += integral_b[:, 0]
    b[quadrangles[boundary, 1]] += integral_b[:, 1]
    b[quadrangles[boundary, 2]] += integral_b[:, 2]
    #print(b)

    row_boundary = np.repeat(quadrangles[boundary, ::2], 2)
    col_boundary = np.tile(quadrangles[boundary, ::2], 2)
    data_boundary = integral_kefs

    #for i in quadrangles[boundary, 0]

    # надо еще для граничных делоне краевую часть сделать, по половинкам которая 
    # оно зависит тока от граничного в узле * пол ребра
    # это будут числа - можно просто в вектор правой части для определенных нодов
    # надо в ноды добавить числа

    half = (e_D_lenghts[boundary] / 2).reshape(-1)
    q_V0 = bc(*coords[quadrangles[boundary, 0]].T, *e_Vs[boundary].T) * half
    q_V2 = bc(*coords[quadrangles[boundary, 2]].T, *e_Vs[boundary].T) * half

    b[quadrangles[boundary, 0]] += q_V0
    b[quadrangles[boundary, 2]] += q_V2

    # b[quadrangles[boundary, 0]] += q_V * half
    # b[quadrangles[boundary, 2]] += q_V * half

    # Почему тут плюсы а не минусы? Должны быть минусы, но -див дает еще минус и в итоге +
    #exit()

    row = np.concatenate((row_inner, row_boundary))
    col = np.concatenate((col_inner.flat, col_boundary.flat))
    data = np.concatenate((data_inner.flat, data_boundary.flat))

    return row, col, data, b, Bs, q_coeffs_inner, q_coeffs_boundary, q_b


def approximate_div_q(integral_coeffs, quadrangles, quadrangle_mask_inner, quadrangle_mask_boundary, coords, q_exact):
    # inner
    row = np.repeat(quadrangles[quadrangle_mask_inner], 4)
    col = np.tile(quadrangles[quadrangle_mask_inner], 4)
    data = integral_coeffs

    # boundary

    quadrangles_boundary = quadrangles[quadrangle_mask_boundary]
    quadrangles_boundary_coords = coords[quadrangles_boundary]

    is_first_V_middle = np.all(np.isclose(quadrangles_boundary_coords[:, ::2].sum(axis=1) / 2, quadrangles_boundary_coords[:, 1]), axis=1)
    print(is_first_V_middle.shape)
    print(is_first_V_middle)

    # типо внешняя нормаль
    n_nodes = np.where(is_first_V_middle.reshape(-1, 1), quadrangles_boundary[:, 3::-2], quadrangles_boundary[:, 1::2])

    n_node_coords = coords[n_nodes]
    n = n_node_coords[:, 1] - n_node_coords[:, 0]
    n /= np.linalg.norm(n, axis=1).reshape(-1, 1)

    # 
    print(n_nodes)
    print(n, n.shape)

    f = np.zeros(coords.shape[0])

    e_Ds = quadrangles_boundary_coords[:, 2] - quadrangles_boundary_coords[:, 0]
    e_Vs = quadrangles_boundary_coords[:, 3] - quadrangles_boundary_coords[:, 1]

    e_D_lenghts = np.linalg.norm(e_Ds, axis=1).reshape(-1, 1)
    e_V_lenghts = np.linalg.norm(e_Vs, axis=1).reshape(-1, 1)

    data1, row1, col1 = [], [], []

    # у нас нет нода вороного, значит нельзя использовать интеграл коефс, тк там выражение от 4 вершин
    for quadrangle, n_nodes_i, ni, e_D_lenght, e_V_lenght in zip(quadrangles_boundary, n_nodes, n, e_D_lenghts, e_V_lenghts):
        # в ni[0] просто пишем граничное ребро * граничное условие на нормаль, это число и его надо писать в вектор ф, мда
        inner_V, boundary_V = n_nodes_i
        q_V = ni @ q_exact(*coords[boundary_V]).reshape(-1) # qn
        integral_V = q_V * e_D_lenght
        f[inner_V] = -integral_V

        # inner part (внутри 4-х угольника)
        # не соблюдаем то, что еV 90 против часовой от eD

        col1.extend(quadrangle[::2])

        coef = (k)
        # тут нет матрицы k

        # в узлы Делоне пишем ноормаль в нем на пол ребра, тоже в вектор ф

        print(q_exact(*coords[boundary_V]).shape, ni.shape, (ni @ q_exact(*coords[boundary_V])).shape, f[inner_V])

        print(e_D_lenght, e_V_lenght)
        exit()

    return row.flat, col.flat, data.flat


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


def calculate(quadrangle_mesh_name, k, info=False, plot=False):
    x, y, nx, ny = sympy.symbols('x y nx ny')
    u = sympy.exp(x*y) # x * (x - 1) * sympy.sin(4 * sympy.pi * y / 3)
    r = 1

    # - div (k * grad u) + r*u = f
    n = sympy.Matrix([nx, ny])
    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], -div + r * u, "numpy")
    u_exact = sympy.lambdify([x, y], u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], q.dot(n), "numpy")
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

    row, col, data, b_div, Bs, q_coeffs_inner, q_coeffs_boundary, q_b = create_local_coordinate_systems(k, quadrangles, coords, inner, boundary, bc)

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
    print(tmp.min(), tmp.max())

    # np.savetxt('A_Neumann.txt', A.toarray(), fmt='%5.2f')
    # np.savetxt('A_Neumann_dif.txt', tmp.toarray(), fmt='%5.2f')
    # print(groups)

    b = f(coords[:groups[2], 0], coords[:groups[2], 1]) * areas[:groups[2]] + b_div[:groups[2]]

    t.append(time.time())
    u = spsolve(A, b)
    t.append(time.time())

    L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = compute_errors(u, ue, areas, groups)

    quadrangle_centers = utility_mvd.compute_intersection_points(coords[quadrangles])
    q_e = q_exact(quadrangle_centers[:, 0], quadrangle_centers[:, 1]).T.reshape(-1, 2)
    #print(q_b.shape)
    q_local_inner = np.sum(q_coeffs_inner * u[quadrangles[inner]].reshape(-1, 1, 4), axis=2)
    q_local_boundary = np.sum(q_coeffs_boundary * u[quadrangles[boundary, ::2]].reshape(-1, 1, 2), axis=2) - q_b

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
    print(f'Time\ttotal {time_total:.2f}')
    print(f'\tload  {time_load:.2f} ({time_load / time_total * 100:.2f}%)')
    print(f'\tsolve {time_solve:.2f} ({time_solve / time_total * 100:.2f}%)')
    print(f'\tother {time_other:.2f} ({time_other / time_total * 100:.2f}%)')

    return coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q



if __name__ == '__main__':
    k = np.array(
        ((1, 0.0),
         (0.0, 1))
    )
    res = calculate('meshes/rectangle/rectangle_21_quadrangle', k, info=True, plot=False)
    print(*res)

    # Если к симметричная, то матрица симметричная, а если нет, то нет
    # В дирихле при любой к симметричная (после лифтинга)

    # 193801 1.9562637730867953e-07 1.7184392399539124e-07 2.603843576581582e-07 2.0100485735863316e-06 2.3723977249368033e-06 2.3723977249368033e-06 2.5126142249882246e-05
    # 193801 6.146895727608258e-06  9.119710596786576e-06  1.099788381713637e-05 4.059213710538856e-05  1.439699248040327e-05  4.059213710538856e-05  3.115950781353237e-05