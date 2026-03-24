import numpy as np
import sympy
import utility_mvd
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import time
import pathlib


def assemble_div_q(k, quadrangles, quadangle_coords, quadrangle_centers, inner, boundary, bc_f, c_f, nodes, groups):
    eD = quadangle_coords[:, 2] - quadangle_coords[:, 0]
    eV = quadangle_coords[:, 3] - quadangle_coords[:, 1]
    lD = np.linalg.norm(eD, axis=1)
    lV = np.linalg.norm(eV, axis=1)
    eD /= lD.reshape(-1, 1)
    eV /= lV.reshape(-1, 1)

    quadrangle_areas = lD * lV / 2

    B = np.stack((eD, eV), axis=-1)
    k = k(*quadrangle_centers.T)
    k = k if k.ndim == 2 else np.transpose(k, axes=(2, 0, 1))
    k = np.linalg.inv(B) @ k @ B

    g_matrix_inner = k[inner]
    g_matrix_inner[:, :, 0] /= lD[inner].reshape(-1, 1)
    g_matrix_inner[:, :, 1] /= lV[inner].reshape(-1, 1)
    g_matrix_inner = np.tile(g_matrix_inner, 2)
    g_matrix_inner[:, :, 2:] *= -1

    kDD = k[boundary, 0, 0]
    kDV = k[boundary, 0, 1]
    kVD = k[boundary, 1, 0]
    kVV = k[boundary, 1, 1]
    a = b = 0.5
    c = c_f(*quadrangle_centers[boundary].T)
    bc = bc_f(*quadrangle_centers[boundary].T, *eV[boundary].T)
    j = kVV + 2*b*c*lV[boundary]

    g_matrix_boundary = np.zeros((quadrangles[boundary].shape[0], 2, 5))
    g_matrix_boundary[:, 0, 0] = kDD/lD[boundary] - kDV*kVD/(lD[boundary]*j)
    g_matrix_boundary[:, 0, 1] = c * kDV * (a+b) / j
    g_matrix_boundary[:, 0, 2] -= g_matrix_boundary[:, 0, 0]
    g_matrix_boundary[:, 0, 4] = bc * kDV / j
    g_matrix_boundary[:, 1, 0] = 2 * b * c * lV[boundary] * kVD / (lD[boundary] * j)
    g_matrix_boundary[:, 1, 1] = c * kVV * (a + b) / j
    g_matrix_boundary[:, 1, 2] -= g_matrix_boundary[:, 1, 0]
    g_matrix_boundary[:, 1, 4] = bc * kVV / j

    g_matrix1 = np.concatenate((np.pad(g_matrix_inner, ((0,0), (0,0), (0,1))), g_matrix_boundary))
    g_matrix = np.zeros((quadrangles.shape[0], 2, 5))
    g_matrix[inner, :, :4] = g_matrix_inner
    g_matrix[boundary] = g_matrix_boundary
    print(np.allclose(g_matrix, g_matrix1))
    exit()

    integral_matrix = g_matrix.copy()
    integral_matrix[:, 0] *= lV.reshape(-1, 1)
    integral_matrix[:, 1] *= lD.reshape(-1, 1)
    integral_matrix = np.tile(integral_matrix, (1, 2, 1))
    integral_matrix[:, 2:] *= -1

    half_lD = lD[boundary] / 2
    integral_matrix[boundary, 0, 0] += half_lD * c_f(*nodes[quadrangles[boundary, 0]].T)
    integral_matrix[boundary, 0, 4] += half_lD * bc_f(*nodes[quadrangles[boundary, 0]].T, *eV[boundary].T)
    integral_matrix[boundary, 2, 2] += half_lD * c_f(*nodes[quadrangles[boundary, 2]].T)
    integral_matrix[boundary, 2, 4] += half_lD * bc_f(*nodes[quadrangles[boundary, 2]].T, *eV[boundary].T)

    row = np.repeat(quadrangles, 4)
    col = np.tile(quadrangles, 4).flatten()
    data = integral_matrix[:, :, :4].flatten()

    b = np.zeros(groups[3])
    b[quadrangles[boundary, 0]] -= integral_matrix[boundary, 0, 4]
    b[quadrangles[boundary, 1]] -= integral_matrix[boundary, 1, 4]
    b[quadrangles[boundary, 2]] -= integral_matrix[boundary, 2, 4]

    g_vector = g_matrix[:, :, 4]
    g_matrix = g_matrix[:, :, :4]

    return row, col, data, b, B, g_matrix, g_vector, quadrangle_areas


def create_local_coordinate_systems(k, quadrangles, coords):
    quadangles_coords = coords[quadrangles]

    e_Ds = quadangles_coords[:, 2] - quadangles_coords[:, 0]
    e_Vs = quadangles_coords[:, 3] - quadangles_coords[:, 1]

    e_D_lenghts = np.linalg.norm(e_Ds, axis=1).reshape(-1, 1)
    e_V_lenghts = np.linalg.norm(e_Vs, axis=1).reshape(-1, 1)
    
    e_Ds /= e_D_lenghts
    e_Vs /= e_V_lenghts

    Bs = np.stack((e_Ds, e_Vs), axis=-1)

    ks = np.linalg.inv(Bs) @ k @ Bs

    q_coeffs = ks
    q_coeffs[:, :, 0] /= e_D_lenghts
    q_coeffs[:, :, 1] /= e_V_lenghts
    q_coeffs = np.tile(q_coeffs, 2)
    q_coeffs[:, :, :2] *= -1

    integral_coeffs = q_coeffs.copy()
    integral_coeffs *= -1
    integral_coeffs[:, 0] *= e_V_lenghts
    integral_coeffs[:, 1] *= e_D_lenghts
    integral_coeffs = np.tile(integral_coeffs, (1, 2, 1))
    integral_coeffs[:, 2:] *= -1

    return Bs, q_coeffs, integral_coeffs


def approximate_div_q(integral_coeffs, quadrangles):
    row = np.repeat(quadrangles, 4)
    col = np.tile(quadrangles, 4)
    data = integral_coeffs
    return row.flat, col.flat, data.flat





def calculate(quadrangle_mesh_name, k, info=False, plot=False):
    k, r, f, bc, u, q = setup_problem()

    t = [time.time()]
    coords, quadrangles, groups, cells, quadrangle_areas = load_data(quadrangle_mesh_name)
    t.append(time.time())

    u_e = u_exact(coords[:, 0], coords[:, 1])

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

    Bs, q_coeffs, integral_coeffs = create_local_coordinate_systems(k, quadrangles, coords)

    row, col, data = approximate_div_q(integral_coeffs, quadrangles)
    row_r, col_r, data_r = approximate_r_u(r, areas, coords[:groups[2]])

    row = np.concatenate((row, row_r))
    col = np.concatenate((col, col_r))
    data = np.concatenate((data, data_r))

    A = coo_array((data, (row, col)), shape=(coords.shape[0], coords.shape[0]))
    A.eliminate_zeros()
    A = A.tocsr()

    b = np.concatenate((
        f(coords[:groups[1], 0], coords[:groups[1], 1]) * areas[:groups[1]],
        u_e[groups[1]:]
    ))

    # lifting
    b[:groups[1]] -= A[:groups[1], groups[1]:] @ b[groups[1]:]
    A.resize((groups[1], groups[1]))
    b.resize(groups[1])

    # check symmetry
    tmp = A - A.T
    print(f'Symmetry {tmp.min()} {tmp.max()}')

    if A.shape[0] < 1e5:
        tmp = A.todense()
        check1 = np.allclose(tmp[:groups[1], groups[1]:groups[2]], 0)
        check2 = np.allclose(tmp[groups[2]:groups[3], groups[1]:groups[2]], 0)
        check3 = np.allclose(tmp[groups[1]:groups[2], :groups[1]], 0)
        check4 = np.allclose(tmp[groups[1]:groups[2], groups[2]:groups[3]], 0)

        check = (check1, check2, check3, check4)

        print(f'Check separation D&V {np.all(check)} {check}')

    t.append(time.time())
    u = spsolve(A, b)
    t.append(time.time())

    u = np.concatenate((u, u_e[groups[1]:]))

    L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = compute_errors(u, u_e, areas, groups)

    quadrangle_centers = utility_mvd.compute_intersection_points(coords[quadrangles])
    q_e = q_exact(quadrangle_centers[:, 0], quadrangle_centers[:, 1]).T.reshape(-1, 2)

    q_local = np.sum(q_coeffs * u[quadrangles].reshape(-1, 1, 4), axis=2)
    q = np.reshape(Bs @ q_local.reshape(-1, 2, 1), (-1, 2))

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

    print(u_e.max(), u.max())

    return coords.shape[0], L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q


def setup_problem():
    x, y = sympy.symbols('x y')

    u = sympy.exp(x*y)
    k = sympy.Matrix([
        [1, 0.0],
        [0.0, 2]
    ])
    r = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y], u, "numpy")

    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, bc, u, q


if __name__ == '__main__':
    k = np.array(
        ((1, 0.2),
         (0.2, 3))
    )
    res = calculate('meshes/rectangle/rectangle_7_quadrangle', k, info=True, plot=False)
    print(*res)
