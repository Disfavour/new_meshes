import gmsh
import numpy as np
import sympy
from scipy.sparse import coo_array, csr_array
from scipy.sparse.linalg import spsolve, inv
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
def calculate(quadrangle_mesh_name, k, info=False, plot=False):
    x, y = sympy.symbols('x y')

    u_exact = sympy.exp(x*y)
    #u_exact = x * (x - 1) * sympy.sin(4 * sympy.pi * y / 3)
    c = 0

    grad_u = sympy.Matrix([u_exact.diff(x), u_exact.diff(y)])
    flux = k * grad_u
    div_flux = flux[0].diff(x) + flux[1].diff(y)
    f = -div_flux + c * u_exact

    u_exact = sympy.lambdify([x, y], u_exact, "numpy")
    f = sympy.lambdify([x, y], f, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")
    grad = sympy.lambdify([x, y], grad_u, "numpy")

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
    
    v_col, v_data, gammas = calculate_vector_values(quad_nodes, node_coords, k)

    gamma = np.abs(gammas).max()
    tau = 2 / (1 + gamma)
    print(gamma, tau)

    return node_coords[quad_nodes], gammas


if __name__ == '__main__':
    sigma = 0
    k = np.array(
        ((1, sigma),
         (sigma, 100))
    )
    res = calculate('meshes/rectangle/rectangle_9_quadrangle', k, 2, 1, info=True, plot=False)
    #print(res)

    # матрица симметрична при любой к, ответ меняется на 1е-12 при лифтинге
