import numpy as np
import sympy
import utility_mvd
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import time
import pathlib


def solve(k, r, f_f, c, bc_f, u_f, q_f, ts, sigma, quadrangle_mesh_fname, info=False):
    '''Тут только u зависит от t, следовательно, f, q тоже'''
    t_w = [time.time()]

    nodes, quadrangles = utility_mvd.load_msh(quadrangle_mesh_fname)
    t_w.append(time.time())
    print(f'load_mesh {t_w[-1] - t_w[-2]}')

    groups, cells = utility_mvd.load_npz(pathlib.Path(quadrangle_mesh_fname).with_suffix('.npz'))
    t_w.append(time.time())
    print(f'load_npz {t_w[-1] - t_w[-2]}')

    cell_areas = utility_mvd.compute_cell_areas(cells, groups, nodes)

    boundary = (quadrangles[:, 0] >= groups[2]) & (quadrangles[:, 2] >= groups[2])
    inner = ~boundary

    utility_mvd.redirect_eV_on_boundary(quadrangles, nodes, boundary)

    quadrangle_centers = utility_mvd.compute_intersection_points(*nodes[quadrangles].T)

    eD, eV, lD, lV = utility_mvd.compute_basis_vectors_and_lenghts(quadrangles, nodes)
    quadrangle_areas = utility_mvd.compute_quadrangle_areas(lD, lV)

    B = utility_mvd.assemble_change_of_basis_matrices(eD, eV)
    k = utility_mvd.compute_local_k(k, B)

    t_w.append(time.time())
    print(f'preparations {t_w[-1] - t_w[-2]}')

    tau = ts[1] - ts[0]

    inner_matrix_g = utility_mvd.assemble_matrix_g(k[inner], lD[inner], lV[inner])
    boundary_matrix_g = utility_mvd.assemble_boundary_matrix_g(k[boundary], lD[boundary], lV[boundary], c)
    matrix_g = utility_mvd.join_matrix_g(inner_matrix_g, boundary_matrix_g, inner, boundary)

    integral_matrix = utility_mvd.assemble_integral_matrix(matrix_g, lD, lV)
    integral_matrix = utility_mvd.add_boundary_integrals_to_matrix(integral_matrix, lD, boundary, c, c)

    row, col, data = utility_mvd.assemble_coo_sparse_format(quadrangles, integral_matrix)
    row_r, col_r, data_r = utility_mvd.assemble_ru(r, cell_areas)
    #row_t, col_t, data_t, b_t = utility_mvd.assemble_du_dt(tau, yn, cell_areas)
    row_t, col_t, data_t = utility_mvd.assemble_ru(1/tau, cell_areas)

    row = np.concatenate((row, row_r))
    col = np.concatenate((col, col_r))
    data = np.concatenate((data, data_r))

    A = coo_array((data, (row, col)), shape=(groups[4], groups[4]))
    A.eliminate_zeros()
    A = A.tocsr()
    A.resize((groups[3], groups[3]))

    At = coo_array((data_t, (row_t, col_t)), shape=(groups[3], groups[3]))
    At.eliminate_zeros()
    At = At.tocsr()

    An = (1 - sigma) * A
    A = sigma * A + At

    t = ts[0]

    f = f_f(*nodes[:groups[3]].T, t)
    bc0, bc2, bc3 = bc_f(*nodes[quadrangles[boundary, 0]].T, *eV[boundary].T, t), bc_f(*nodes[quadrangles[boundary, 2]].T, *eV[boundary].T, t), bc_f(*nodes[quadrangles[boundary, 3]].T, *eV[boundary].T, t)

    boundary_vector_g = utility_mvd.assemble_boundary_vector_g(k[boundary], lV[boundary], c, bc3)
    boundary_integral_vector = utility_mvd.assemble_boundary_integral_vector(boundary_vector_g, lD[boundary], lV[boundary])
    boundary_integral_vector = utility_mvd.add_boundary_integrals_to_vector(boundary_integral_vector, lD[boundary], bc0, bc2)
    b_div = utility_mvd.assemble_vector_b(boundary_integral_vector, quadrangles, boundary, groups[3])
    b_f = f * cell_areas
    y = u_f(*nodes[:groups[3]].T, t)

    norms = []
    def collect_data(i):
        u = u_f(*nodes[:groups[3]].T, t)
        L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = utility_mvd.compute_errors(y, u, cell_areas, groups)
    
        q = np.squeeze(q_f(*quadrangle_centers.T, t)).T
        g = utility_mvd.compute_global_g(B, matrix_g, boundary_vector_g, y, quadrangles, boundary)
        L2_q = utility_mvd.compute_vector_error(g, q, quadrangle_areas)

        norms.append((L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax, L2_q))

        t_w.append(time.time())
        print(f'{i:{len(str(ts.size - 1))}}/{ts.size - 1} {t:.3f}/{ts[-1]:.3f} iter time {t_w[-1] - t_w[-2]:7.2f}\t' \
              + f'L2_D {L2_D:.3e} L2_V {L2_V:.3e} L2 {L2:.3e}\tLmax_D {Lmax_D:.3e} Lmax_V {Lmax_V:.3e} Lmax {Lmax:.3e}\tL2_q {L2_q:.3e}')

    collect_data(0)
    yn = y
    b_div_n = b_div
    b_f_n = b_f

    for i, t in enumerate(ts[1:], 1):
        f = f_f(*nodes[:groups[3]].T, t)
        bc0, bc2, bc3 = bc_f(*nodes[quadrangles[boundary, 0]].T, *eV[boundary].T, t), bc_f(*nodes[quadrangles[boundary, 2]].T, *eV[boundary].T, t), bc_f(*nodes[quadrangles[boundary, 3]].T, *eV[boundary].T, t)

        boundary_vector_g = utility_mvd.assemble_boundary_vector_g(k[boundary], lV[boundary], c, bc3)
        boundary_integral_vector = utility_mvd.assemble_boundary_integral_vector(boundary_vector_g, lD[boundary], lV[boundary])
        boundary_integral_vector = utility_mvd.add_boundary_integrals_to_vector(boundary_integral_vector, lD[boundary], bc0, bc2)
        b_div = utility_mvd.assemble_vector_b(boundary_integral_vector, quadrangles, boundary, groups[3])
        b_f = f * cell_areas
        bt = 1 / tau * cell_areas * yn

        b = sigma * b_div - An @ yn + (1 - sigma)*b_div_n

        y = spsolve(A, b + sigma*b_f + (1 - sigma)*b_f_n + bt)

        collect_data(i)
        yn = y
        b_div_n = b_div
        b_f_n = b_f

    return nodes.shape[0], np.array(norms)


def setup_problem():
    # div (-k * grad u) + r*u = f
    # q*n - cu = bc
    x, y, nx, ny, t = sympy.symbols('x y nx ny t')
    n = sympy.Matrix([nx, ny])

    k = np.array((
        (1, 3),
        (3, 10)
    ))

    u = sympy.exp(-(1 + 10*t*x)*y**2)
    
    r = 1
    c = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f_f = sympy.lambdify([x, y, t], u.diff(t) + div + r*u, "numpy")
    bc_f = sympy.lambdify([x, y, nx, ny, t], -q.dot(n) + c*u, "numpy")

    u_f = sympy.lambdify([x, y, t], u, "numpy")
    q_f = sympy.lambdify([x, y, t], q, "numpy")

    return k, r, f_f, c, bc_f, u_f, q_f


if __name__ == '__main__':
    ts = np.linspace(0, 1, 21)
    res = solve(*setup_problem(), ts, 0.5, 'meshes/rectangle/rectangle_7_quadrangle.msh', info=True)
    #print(*res)

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