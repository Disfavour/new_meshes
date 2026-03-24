import numpy as np
import sympy
import utility_mvd
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import time
import pathlib


def assemble_div_q(k, quadrangles, quadrangle_centers, nodes):
    eD, eV, lD, lV = utility_mvd.compute_basis_vectors_and_lenghts(quadrangles, nodes)
    quadrangle_areas = utility_mvd.compute_quadrangle_areas(lD, lV)
    
    B = utility_mvd.assemble_change_of_basis_matrices(eD, eV)
    global_k = np.transpose(np.atleast_3d(k(*quadrangle_centers.T)), axes=(2, 0, 1))
    k = utility_mvd.compute_local_k(global_k, B)

    matrix_g = utility_mvd.assemble_matrix_g(k, lD, lV)
    
    integral_matrix = utility_mvd.assemble_integral_matrix(matrix_g.copy(), lD, lV)

    row, col, data = utility_mvd.assemble_coo_sparse_format(quadrangles, integral_matrix)

    return row, col, data, B, matrix_g, quadrangle_areas


def solve_problem(k, r, f, u, q, quadrangle_mesh_fname, info=False):
    t = [time.time()]

    nodes, quadrangles = utility_mvd.load_msh(quadrangle_mesh_fname)
    #t.append(time.time())

    groups, cells = utility_mvd.load_npz(pathlib.Path(quadrangle_mesh_fname).with_suffix('.npz'))
    t.append(time.time())

    u = u(*nodes[:groups[4]].T)

    cell_areas = utility_mvd.compute_cell_areas(cells, groups, nodes)

    quadrangle_centers = utility_mvd.compute_intersection_points(*nodes[quadrangles].T)

    row, col, data, B, matrix_g, quadrangle_areas = assemble_div_q(k, quadrangles, quadrangle_centers, nodes)
    
    row_r, col_r, data_r = utility_mvd.assemble_ru(r, cell_areas, nodes[:groups[3]])

    row = np.concatenate((row, row_r))
    col = np.concatenate((col, col_r))
    data = np.concatenate((data, data_r))

    A = coo_array((data, (row, col)), shape=(groups[4], groups[4]))
    A.eliminate_zeros()
    A = A.tocsr()
    b = np.concatenate((
        f(nodes[:groups[2], 0], nodes[:groups[2], 1]) * cell_areas[:groups[2]],
        u[groups[2]:]
    ))

    # lifting
    b[:groups[2]] -= A[:groups[2], groups[2]:] @ b[groups[2]:]
    A.resize((groups[2], groups[2]))
    b.resize(groups[2])

    if info:
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

    # np.savetxt('A_matrix_new.txt', A.toarray(), fmt='%5.2f')
    # print(groups)

    t.append(time.time())
    y = spsolve(A, b)
    t.append(time.time())

    y = np.concatenate((y, u[groups[2]:]))

    L2_D, L2_V, L2, Lmax_D, Lmax_V, Lmax = utility_mvd.compute_errors(y, u, cell_areas, groups)
    
    q = np.squeeze(q(*quadrangle_centers.T)).T
    g = utility_mvd.compute_global_g(B, matrix_g, None, y, quadrangles)
    L2_q = utility_mvd.compute_vector_error(g, q, quadrangle_areas)
    
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
    x, y = sympy.symbols('x y')

    u = sympy.exp(x*y)
    k = sympy.Matrix([
        [1, 3],
        [3, 10]
    ])
    r = 1

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -k * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    k = sympy.lambdify([x, y], k, "numpy")
    r = sympy.lambdify([x, y], r, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")
    q = sympy.lambdify([x, y], q, "numpy")

    return k, r, f, u, q


if __name__ == '__main__':
    res = solve_problem(*setup_problem(), 'meshes/rectangle/rectangle_7_quadrangle.msh', info=True)
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