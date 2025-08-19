import gmsh
import numpy as np
import sympy
import sys
#sys.path.append('mesh_generation')
#import utility
import utility_mvd
from scipy.sparse import lil_array, coo_array
from scipy.sparse.linalg import spsolve


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
    row, col, data  = [], [], []

    B = np.column_stack((
        (1, 0),
        (0, 1)
    ))

    vector_values = []
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

        grad = np.array((
            np.zeros(node_coords.shape[0]),
            np.zeros(node_coords.shape[0])
        ))

        grad[0][quad_nodes[2]] = 1
        grad[0][quad_nodes[0]] = -1
        grad[0] /= e_D_lenght

        grad[1][quad_nodes[3]] = 1
        grad[1][quad_nodes[1]] = -1
        grad[1] /= e_V_lenght

        vector_values.append(k_new_basis @ grad)        

        row.extend((
            i,
            i,
            i,
            i,
            i,
            i,
            i,
            i,
        ))
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


    return np.array(vector_values), np.array(row), np.array(col), np.array(data)


def assemble_matrix_for_inner_nodes(inner_nodes, cell_nodes, quad_nodes, vector_values, node_coords, cell_areas, c, v_row, v_col, v_data, Voronoi=False):
    row, col, data = [], [], []
    matrix_A = []
    for inner_node, current_cell_nodes, cell_area in zip(inner_nodes, cell_nodes, cell_areas):
        div = 0
        for n1, n2 in zip(current_cell_nodes, np.roll(current_cell_nodes, -1)):
            # Сделать через ребра
            quad_index = (((quad_nodes == n1) + (quad_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item()
            current_quad_nodes = quad_nodes[quad_index]

            vector_normal_component = vector_values[quad_index][0 + Voronoi]
            if inner_node != current_quad_nodes[0 + Voronoi]:
                vector_normal_component = -vector_normal_component
            
            # v_row = quad_index
            # vn = vector_values[quad_index][0 + Voronoi]
            # if inner_node != current_quad_nodes[0 + Voronoi]:
            #     vector_normal_component = -vector_normal_component
                                    
            edge_lenght = np.linalg.norm((node_coords[n1] - node_coords[n2]))
            
            div += vector_normal_component * edge_lenght


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

        matrix_row = -div # / cell_area (изменил, чтобы матрица была симметричной)
        matrix_row[inner_node] += c(*node_coords[inner_node]) * cell_area

        matrix_A.append(matrix_row)
    
    return np.array(matrix_A), row, col, data #np.array(row), np.array(col), np.array(data)


# - div (k * grad u) + c*u = f
def calculate(quadrangle_mesh_name, k, plot=False):
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

    node_coords, quad_nodes, node_groups, cell_nodes, quad_areas = load_data(quadrangle_mesh_name)
    cell_areas = utility_mvd.calculate_cell_areas(cell_nodes, node_coords)

    row, col, data  = [], [], []
    
    

    vector_values, v_row, v_col, v_data = calculate_vector_values(quad_nodes, node_coords, k)

    A_D_inner, row_D, col_D, data_D = assemble_matrix_for_inner_nodes(
        range(node_groups[0]), cell_nodes[:node_groups[0]], quad_nodes, vector_values, node_coords, cell_areas[:node_groups[0]], c, v_row, v_col, v_data)
    
    A_V_inner, row_V, col_V, data_V = assemble_matrix_for_inner_nodes(
        range(node_groups[0], node_groups[1]), cell_nodes[node_groups[0]:node_groups[1]], quad_nodes, vector_values, node_coords, cell_areas[node_groups[0]:node_groups[1]], c, v_row, v_col, v_data, Voronoi=True)
    
    arr_D = coo_array((data_D, (row_D, col_D)), shape=(node_groups[0], node_coords.shape[0]))

    arr_V = coo_array((data_V, (row_V, col_V)), shape=(node_groups[1], node_coords.shape[0]))

    row = row_D + row_V
    col = col_D + col_V
    data = data_D + data_V

    A_inner_sparse = coo_array((data, (row, col)), shape=(node_groups[1], node_coords.shape[0]))

    
    
    A_inner = np.concatenate((A_D_inner, A_V_inner))
    f_inner = f(node_coords[:node_groups[1], 0], node_coords[:node_groups[1], 1]) * cell_areas[:node_groups[1]]

    A_inner_sparse.eliminate_zeros()
    A_inner_csr = A_inner_sparse.tocsr()

    #print((A_inner[:, :node_groups[1]] - A_inner[:, :node_groups[1]].T).max())

    A_boundary = np.eye(node_groups[3] - node_groups[1], node_coords.shape[0], node_groups[1])
    f_boundary = u_exact(node_coords[node_groups[1]:, 0], node_coords[node_groups[1]:, 1])
    # print(f_boundary)
    # print(f_boundary.min(), f_boundary.max())
    # exit()

    A = np.concatenate((A_inner, A_boundary))
    # f = np.concatenate((f_inner, f_boundary))
    # u_1 = np.linalg.solve(A, f)

    # lifting
    A__inner_bc = A_inner[:, node_groups[1]:]
    A_inner = A_inner[:, :node_groups[1]]
    assert np.allclose(A_inner, A_inner.T)


    f_b_sparse = f_boundary.copy()
    f_i_sparse = f_inner.copy()
    #print(A_inner_csr[:, node_groups[1]:])
    
    #print(A_inner_csr[:, node_groups[1]:])
    f_i_sparse -= A_inner_csr[:, node_groups[1]:] @ f_b_sparse
    A_inner_csr.resize((node_groups[1], node_groups[1]))
    #A_inner_csr[:, node_groups[1]:] = 0
    #A_inner_csr.eliminate_zeros()
    #print(type(A_inner_csr))
    u_sparse = spsolve(A_inner_csr, f_i_sparse)

    f_inner1 = f_inner.copy()
    f_inner = f_inner - A__inner_bc @ f_boundary


    u = np.linalg.solve(A_inner, f_inner)
    print(u)
    print(u_sparse)
    print(np.array_equal(u_sparse, u), np.allclose(u_sparse, u))
    u = np.concatenate((u, f_boundary))
    #assert np.allclose(u, u_1)

    
    exit()

    u_e = u_exact(node_coords[:, 0], node_coords[:, 1])

    L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2 = utility_mvd.calculate_errornorms(u, u_e, cell_areas, node_groups)
    vector_errornorm = utility_mvd.calculate_vector_errornorm(quad_nodes, quad_areas, node_coords, grad, u)

    if plot:
        utility_mvd.plot_results(u, u_e, node_coords, node_groups)
    
    u = np.concatenate((u[:node_groups[0]], u[node_groups[1]:node_groups[2]], u[node_groups[0]:node_groups[1]], u[node_groups[2]:node_groups[3]]))

    f = np.concatenate((f_inner1, f_boundary))
    f = np.concatenate((f[:node_groups[0]], f[node_groups[1]:node_groups[2]], f[node_groups[0]:node_groups[1]], f[node_groups[2]:node_groups[3]]))

    A_DD = A[:node_groups[0], :node_groups[0]]
    A_DV = A[:node_groups[0], node_groups[0]:node_groups[1]]
    A_DpD = A[:node_groups[0], node_groups[1]:node_groups[2]]
    A_DpV = A[:node_groups[0], node_groups[2]:node_groups[3]]

    A_D = np.concatenate((A_DD, A_DpD, A_DV, A_DpV), axis=1)

    A_VD = A[node_groups[0]:node_groups[1], :node_groups[0]]
    A_VV = A[node_groups[0]:node_groups[1], node_groups[0]:node_groups[1]]
    A_VpD = A[node_groups[0]:node_groups[1], node_groups[1]:node_groups[2]]
    A_VpV = A[node_groups[0]:node_groups[1], node_groups[2]:node_groups[3]]

    A_V = np.concatenate((A_VD, A_VpD, A_VV, A_VpV), axis=1)

    A = np.concatenate((A_D, np.eye(node_groups[2] - node_groups[1], node_groups[3], node_groups[0]), A_V, np.eye(node_groups[3] - node_groups[2], node_groups[3], node_groups[2])))

    u1 = np.linalg.solve(A, f)

    #np.savetxt('test_A_inner.txt', A_inner, fmt='%+0.2f')
    return np.array((L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2)), u, f, A, u1
    #return node_coords.shape[0], L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2, vector_errornorm


if __name__ == '__main__':
    k = np.array(
        ((1, 0.5),
         (0, 1))
    )
    res = calculate('meshes/rectangle/rectangle_0_quadrangle', k, plot=False)
    print(res)

    # матрица симметрична при любой к, ответ меняется на 1е-12 при лифтинге
