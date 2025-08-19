import gmsh
import numpy as np
import sympy
import sys
#sys.path.append('mesh_generation')
#import utility
import utility_mvd
from scipy.sparse import lil_array, coo_array
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


# Векторные величины в точках пересечения диагоналей четырехугольных ячеек
def calculate_vector_values(quadrangle_nodes, node_coords, k):
    col, data  = [], []

    B = np.column_stack((
        (1, 0),
        (0, 1)
    ))

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


    return np.array(col), np.array(data)


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
def calculate(quadrangle_mesh_name, k, info=False, plot=False):
    x, y = sympy.symbols('x y')

    u_exact = sympy.exp(x*y)
    #u_exact = x * (x - 1) * sympy.sin(4 * sympy.pi * y / 3)
    c = 0

    grad_u = sympy.Matrix([u_exact.diff(x), u_exact.diff(y)])
    flux = k * grad_u
    div_flux = flux[0].diff(x) + flux[1].diff(y)
    f = -div_flux + c * u_exact
    # print(f)
    # exit()

    u_exact = sympy.lambdify([x, y], u_exact, "numpy")
    f = sympy.lambdify([x, y], f, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    grad = sympy.lambdify([x, y], grad_u, "numpy")

    t = []
    t.append(time.time())
    # 0
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

    t.append(time.time())

    quad_node_coords = np.concatenate((node_coords[quad_nodes][:, ::2], node_coords[quad_nodes][:, 1::2]), axis=1)
    quad_centers = utility_mvd.compute_intersection_points(*quad_node_coords.transpose(1, 2, 0).reshape(-1, quad_nodes.shape[0]))

    t.append(time.time())

    # 1
    #print(cell_nodes.dtype, cell_nodes[0].dtype)
    #print(node_coords[cell_nodes.astype(np.int64)])
    d = {}
    for n in cell_nodes:
        sh = n.size
        if sh not in d:
            d[sh] = 0
        d[sh] += 1
    
    for k, v in d.items():
        print(k, ' -> ', v)
    #exit()

    # node_coords[cell_nodes]
    cell_node_coords = np.array([node_coords[nodes] for nodes in cell_nodes], dtype=object)

    a1 = np.ones((2, 2))
    print(a1 @ a1, np.vecdot(a1, a1))

    print(cell_node_coords.shape)
    print()
    exit()

    cell_areas = utility_mvd.compute_polygon_area(cell_nodes, node_coords)
    t.append(time.time())
    # 2
    v_col, v_data = calculate_vector_values(quad_nodes, node_coords, k)
    t.append(time.time())
    # 3
    row_D, col_D, data_D = assemble_matrix_for_inner_nodes(
        range(node_groups[0]), cell_nodes[:node_groups[0]], quad_nodes, node_coords, cell_areas[:node_groups[0]], c, v_col, v_data, node_to_quads_sorted[:node_groups[0]])
    
    t.append(time.time())
    # 4
    row_V, col_V, data_V = assemble_matrix_for_inner_nodes(
        range(node_groups[0], node_groups[1]), cell_nodes[node_groups[0]:node_groups[1]], quad_nodes, node_coords, cell_areas[node_groups[0]:node_groups[1]], c, v_col, v_data, node_to_quads_sorted[node_groups[0]:node_groups[1]], Voronoi=True)
    
    t.append(time.time())
    # 5
    row, col, data  = [], [], []
    row = row_D + row_V
    col = col_D + col_V
    data = data_D + data_V

    A_inner_sparse = coo_array((data, (row, col)), shape=(node_groups[1], node_coords.shape[0]))
    A_inner_sparse.eliminate_zeros()
    A_inner_csr = A_inner_sparse.tocsr()

    t.append(time.time())
    # 6
    f_inner = f(node_coords[:node_groups[1], 0], node_coords[:node_groups[1], 1]) * cell_areas[:node_groups[1]]
    f_boundary = u_exact(node_coords[node_groups[1]:, 0], node_coords[node_groups[1]:, 1])

    t.append(time.time())
    # 7
    # lifting
    f_inner -= A_inner_csr[:, node_groups[1]:] @ f_boundary

    t.append(time.time())
    # 8
    A_inner_csr.resize((node_groups[1], node_groups[1]))

    t.append(time.time())
    # 9
    u = spsolve(A_inner_csr, f_inner)

    t.append(time.time())
    # 10
    u = np.concatenate((u, f_boundary))

    t.append(time.time())
    # 11
    u_e = u_exact(node_coords[:, 0], node_coords[:, 1])

    t.append(time.time())
    # 12
    L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2 = utility_mvd.calculate_errornorms(u, u_e, cell_areas, node_groups)

    t.append(time.time())
    # 13
    vector_errornorm = utility_mvd.calculate_vector_errornorm(quad_nodes, quad_areas, node_coords, quad_centers, grad, u)

    t.append(time.time())

    t = np.array(t)
    if info:
        print(np.roll(t, -1) - t)

    if plot:
        utility_mvd.plot_results(u, u_e, node_coords, node_groups)
    
    #print(u)
    return node_coords.shape[0], L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2, vector_errornorm


if __name__ == '__main__':
    k = np.array(
        ((1, 0.5),
         (0.2, 1))
    )
    res = calculate('meshes/rectangle/rectangle_18_quadrangle', k, info=True, plot=False)
    print(res)

    # матрица симметрична при любой к, ответ меняется на 1е-12 при лифтинге
