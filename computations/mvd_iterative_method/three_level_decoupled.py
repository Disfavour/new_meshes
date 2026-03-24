import gmsh
import numpy as np
import sympy
from scipy.sparse import coo_array, csr_array
from scipy.sparse.linalg import spsolve, inv
import time
import sys
sys.path.append('computations')
import utility_mvd


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


# Векторные величины в точках пересечения диагоналей четырехугольных ячеек
def calculate_vector_values(quadrangle_nodes, node_coords, k):
    col, data  = [], []

    B = np.column_stack((
        (1, 0),
        (0, 1)
    ))

    gammas = []
    for i, quad_nodes in enumerate(quadrangle_nodes):
        e_D = node_coords[quad_nodes[2]] - node_coords[quad_nodes[0]]
        e_D_lenght = np.linalg.norm(e_D)

        e_V = node_coords[quad_nodes[3]] - node_coords[quad_nodes[1]]
        e_V_lenght = np.linalg.norm(e_V)
        
        B_new_basis = np.column_stack((
            e_D / e_D_lenght,
            e_V / e_V_lenght
        ))

        C = np.linalg.inv(B) @ B_new_basis
        k_new_basis = np.linalg.inv(C) @ k @ C

        gammas.append(k_new_basis[0, 1] / np.sqrt(k_new_basis[0, 0] * k_new_basis[1, 1]))

        col.extend((
            quad_nodes[2],
            quad_nodes[0],
            quad_nodes[3],
            quad_nodes[1],
            quad_nodes[2],
            quad_nodes[0],
            quad_nodes[3],
            quad_nodes[1],
        ))
        data.extend((
            k_new_basis[0, 0] / e_D_lenght,
            -k_new_basis[0, 0] / e_D_lenght,
            k_new_basis[0, 1] / e_V_lenght,
            -k_new_basis[0, 1] / e_V_lenght,
            k_new_basis[1, 0] / e_D_lenght,
            -k_new_basis[1, 0] / e_D_lenght,
            k_new_basis[1, 1] / e_V_lenght,
            -k_new_basis[1, 1] / e_V_lenght,
        ))

    return np.array(col), np.array(data), np.array(gammas)


def assemble_matrix_for_inner_nodes(inner_nodes, cell_nodes, quad_nodes, node_coords, cell_areas, c, v_col, v_data, node_to_quads_sorted, Voronoi=False):
    row, col, data = [], [], []
    for inner_node, current_cell_nodes, cell_area, node_to_quads_sorted_i in zip(inner_nodes, cell_nodes, cell_areas, node_to_quads_sorted):
        for n1, n2, quad_index in zip(current_cell_nodes, np.roll(current_cell_nodes, -1), node_to_quads_sorted_i):
            # Сделать через ребра, node to quad nodes
            # знаю 3 нода но надо знать номер квада
            #quad_index = (((quad_nodes == n1) + (quad_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item()
            current_quad_nodes = quad_nodes[quad_index]

            #assert quad_index == quad_index1
                                    
            edge_lenght = np.linalg.norm((node_coords[n1] - node_coords[n2]))

            # div += Попробовать lil, c не работает
            if Voronoi == False:
                # v_row не нужен
                row.extend([inner_node for i in range(4)])
                col.extend(v_col[quad_index*8:quad_index*8 + 4])
                if inner_node == current_quad_nodes[0 + Voronoi]:
                    data.extend(-v_data[quad_index*8:quad_index*8 + 4] * edge_lenght)
                else:
                    data.extend(v_data[quad_index*8:quad_index*8 + 4] * edge_lenght)
            else:
                row.extend([inner_node for i in range(4)])
                col.extend(v_col[quad_index*8 + 4:quad_index*8 + 8])
                if inner_node == current_quad_nodes[0 + Voronoi]:
                    data.extend(-v_data[quad_index*8 + 4:quad_index*8 + 8] * edge_lenght)
                else:
                    data.extend(v_data[quad_index*8 + 4:quad_index*8 + 8] * edge_lenght)

    
    return row, col, data #np.array(row), np.array(col), np.array(data)


# - div (k * grad u) + c*u = f
def calculate(quadrangle_mesh_name, k, max_iter, tau=None, r_rel_min=-1, info=False, plot=False):
    u_bc = 0
    f = 1
    c = 0

    node_coords, quad_nodes, node_groups, cell_nodes, quad_areas = load_data(quadrangle_mesh_name)
    
    # по ноде знаем порядок вершин (против часовой), а хотим еще порядок квадов против часовой
    # надо знать именно номер квада, в жмше в идеале знать
    # t.append(time.time())
    # quad_to_nodes = {i: nodes for i, nodes in zip(range(quad_nodes.shape[0]), quad_nodes)}
    # node_to_quads = utility_mvd.reverse_dict(quad_to_nodes)
    #t.append(time.time())
    node_to_quads = [[] for i in range(node_coords.shape[0])]
    for i, nodes in enumerate(quad_nodes):
        for node in nodes:
            node_to_quads[node].append(i)
    
    #t.append(time.time())
    node_to_quads_sorted = []
    # исключили граничные вороного
    # при генерации квадов мб как-то запоминать порядок и идти по упорядоченным вершинам ячейки.
    for node, current_cell_nodes in zip(range(node_groups[1]), cell_nodes):
        node_to_quads_sorted.append([])
        for n1, n2 in zip(current_cell_nodes, np.roll(current_cell_nodes, -1)):
            quads = node_to_quads[node]
            q_nodes = quad_nodes[quads]
            q_index = quads[(np.isin(q_nodes, [n1, n2]).sum(axis=1) == 2).nonzero()[0].item()]

            #quad_index = (((quad_nodes == n1) + (quad_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item()

            #assert q_index == quad_index

            node_to_quads_sorted[-1].append(q_index)

    #node_to_quads_sorted = np.array(node_to_quads_sorted)

    # Можно еще словарь 2 ноды (сортед) в квад, но лучше при генерации квадов
    #node_to_quads = np.array(node_to_quads)

    quad_node_coords_special = np.concatenate((node_coords[quad_nodes][:, ::2], node_coords[quad_nodes][:, 1::2]), axis=1)
    quad_centers = utility_mvd.compute_intersection_points(*quad_node_coords_special.transpose(1, 2, 0).reshape(-1, quad_nodes.shape[0]))

    cell_areas = np.array([utility_mvd.compute_polygon_area(*cell_node_coords.T) for cell_node_coords in (node_coords[nodes] for nodes in cell_nodes)])
    
    v_col, v_data, gammas = calculate_vector_values(quad_nodes, node_coords, k)
    
    row_D, col_D, data_D = assemble_matrix_for_inner_nodes(
        range(node_groups[0]), cell_nodes[:node_groups[0]], quad_nodes, node_coords, cell_areas[:node_groups[0]], c, v_col, v_data, node_to_quads_sorted[:node_groups[0]])
        
    row_V, col_V, data_V = assemble_matrix_for_inner_nodes(
        range(node_groups[0], node_groups[1]), cell_nodes[node_groups[0]:node_groups[1]], quad_nodes, node_coords, cell_areas[node_groups[0]:node_groups[1]], c, v_col, v_data, node_to_quads_sorted[node_groups[0]:node_groups[1]], Voronoi=True)
        
    row, col, data  = [], [], []
    row = row_D + row_V
    col = col_D + col_V
    data = data_D + data_V

    A_inner_sparse = coo_array((data, (row, col)), shape=(node_groups[1], node_coords.shape[0]))
    A_inner_sparse.eliminate_zeros()
    A_inner_csr = A_inner_sparse.tocsr()

    #f_inner = f(node_coords[:node_groups[1], 0], node_coords[:node_groups[1], 1]) * cell_areas[:node_groups[1]]
    f_inner = np.full_like(node_coords[:node_groups[1], 0], f) * cell_areas[:node_groups[1]]
    f_boundary = np.full_like(node_coords[node_groups[1]:, 0], u_bc)

    # lifting
    f_inner -= A_inner_csr[:, node_groups[1]:] @ f_boundary

    A_inner_csr.resize((node_groups[1], node_groups[1]))

    # u = spsolve(A_inner_csr, f_inner)

    A = A_inner_csr
    f = f_inner
    y = np.zeros(A.shape[0])

    A_DD = csr_array((node_groups[0], node_groups[0]))
    A_VV = csr_array((node_groups[1] - node_groups[0], node_groups[1] - node_groups[0]))

    A_DD = A[:node_groups[0], :node_groups[0]]
    A_VV = A[node_groups[0]:, node_groups[0]:]

    A_DV = csr_array((node_groups[0], node_groups[1] - node_groups[0]))
    A_VD = csr_array((node_groups[1] - node_groups[0], node_groups[0]))

    A_DV = A[:node_groups[0], node_groups[0]:]
    A_VD = A[node_groups[0]:, :node_groups[0]]

    A_DD_inv = inv(A_DD.tocsc()).tocsr()
    A_VV_inv = inv(A_VV.tocsc()).tocsr()

    f_D = f[:node_groups[0]]
    f_V = f[node_groups[0]:]

    y_D = y[:node_groups[0]]
    y_V = y[node_groups[0]:]

    gamma = np.abs(gammas).max()
    gamma_1 = 1 - gamma
    gamma_2 = 1 + gamma
    xi = gamma_1 / gamma_2
    rho_0 = (1 - xi) / (1 + xi)

    if tau is None:
        tau = 2 / (1 + gamma)

    f_norm = np.linalg.norm(f)

    # 0
    r_D = A_DD @ y_D + A_DV @ y_V - f_D
    r_V = A_VD @ y_D + A_VV @ y_V - f_V
    r_abs = [np.linalg.norm(np.concatenate((r_D, r_V)))]
    r_rel = [r_abs[-1] / f_norm]

    if info:
        print(0, r_abs[-1], r_rel[-1])
        
    if (r_rel[-1] < r_rel_min):
        return np.array(r_abs), np.array(r_rel), gamma, tau

    yk_D, yk_V = y_D, y_V

    # 1
    a = 2

    y_D = A_DD_inv @ ((1 - tau)*A_DD @ yk_D - tau*A_DV @ yk_V + tau*f_D)
    y_V = A_VV_inv @ (-tau*A_VD @ yk_D + (1 - tau)*A_VV @ yk_V + tau*f_V)

    r_D = A_DD @ y_D + A_DV @ y_V - f_D
    r_V = A_VD @ y_D + A_VV @ y_V - f_V
    r_abs.append(np.linalg.norm(np.concatenate((r_D, r_V))))
    r_rel.append(r_abs[-1] / f_norm)

    if info:
        print(1, r_abs[-1], r_rel[-1])
        
    if (r_rel[-1] < r_rel_min):
        return np.array(r_abs), np.array(r_rel), gamma, tau

    yk1_D, yk1_V = yk_D, yk_V
    yk_D, yk_V = y_D, y_V

    for i in range(2, max_iter + 1):
        a = 4 / (4 - rho_0**2 * a)

        y_D = A_DD_inv @ (a*((1 - tau)*A_DD @ yk_D - tau*A_DV @ yk_V) + (1 - a)*A_DD @ yk1_D + a*tau*f_D)
        y_V = A_VV_inv @ (a*(-tau*A_VD @ yk_D + (1 - tau)*A_VV @ yk_V) + (1 - a)*A_VV @ yk1_V + a*tau*f_V)

        r_D = A_DD @ y_D + A_DV @ y_V - f_D
        r_V = A_VD @ y_D + A_VV @ y_V - f_V
        r_abs.append(np.linalg.norm(np.concatenate((r_D, r_V))))
        r_rel.append(r_abs[-1] / f_norm)

        if info:
            print(i, r_abs[-1], r_rel[-1])
        
        if r_rel[-1] < r_rel_min:
            break

        yk1_D, yk1_V = yk_D, yk_V
        yk_D, yk_V = y_D, y_V

    return np.array(r_abs), np.array(r_rel), gamma, tau


if __name__ == '__main__':
    sigma = 7
    k = np.array(
        ((1, sigma),
         (sigma, 100))
    )
    res = calculate('meshes/rectangle/rectangle_11_quadrangle', k, 10, 1, info=True, plot=False)
    #print(res)

    # матрица симметрична при любой к, ответ меняется на 1е-12 при лифтинге
