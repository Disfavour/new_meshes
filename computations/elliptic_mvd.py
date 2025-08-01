import gmsh
import numpy as np
import sympy
import sys
#sys.path.append('mesh_generation')
#import utility
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
def calculate_vector_values(quad_nodes, node_coords, k):
    B = np.column_stack((
        (1, 0),
        (0, 1)
    ))

    vector_values = []
    for quad_nodes in quad_nodes:
        e_D = (node_coords[quad_nodes[2]] - node_coords[quad_nodes[0]])
        e_D_lenght = np.linalg.norm(e_D)

        e_V = (node_coords[quad_nodes[3]] - node_coords[quad_nodes[1]])
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
    
    return np.array(vector_values)


def assemble_matrix_and_vector_for_inner_nodes(inner_nodes, cell_nodes, quad_nodes, vector_values, node_coords, cell_areas, c, f, Voronoi=False):
    matrix_A, vector_f = [], []
    for inner_node, current_cell_nodes, cell_area in zip(inner_nodes, cell_nodes, cell_areas):
        div = 0
        for n1, n2 in zip(current_cell_nodes, np.roll(current_cell_nodes, -1)):
            quad_index = (((quad_nodes == n1) + (quad_nodes == n2)).sum(axis=1) == 2).nonzero()[0].item()
            current_quad_nodes = quad_nodes[quad_index]

            vector_normal_component = vector_values[quad_index][0 + Voronoi]
            if inner_node != current_quad_nodes[0 + Voronoi]:
                vector_normal_component = -vector_normal_component
            
            edge_lenght = np.linalg.norm((node_coords[n1] - node_coords[n2]))

            div += vector_normal_component * edge_lenght

        matrix_row = -div # / cell_area (изменил, чтобы матрица была симметричной)
        matrix_row[inner_node] += c(*node_coords[inner_node]) * cell_area

        matrix_A.append(matrix_row)
        vector_f.append(f(*node_coords[inner_node]) * cell_area)
    
    return np.array(matrix_A), np.array(vector_f)


def assemble_matrix_and_vector_for_boundary_nodes(boundary_nodes, node_coords, f):
    matrix_A, vector_f = [], []
    for node in boundary_nodes:
        matrix_row = np.zeros(node_coords.shape[0])
        matrix_row[node] = 1

        matrix_A.append(matrix_row)
        vector_f.append(f(*node_coords[node]))
    
    return np.array(matrix_A), np.array(vector_f)


# - div (k * grad u) + c*u = f
def calculate(quadrangle_mesh_name, k, plot=False):
    x, y = sympy.symbols('x y')

    u_exact = sympy.exp(x*y)# x**2 + x*y #sympy.exp(x*y)
    c = 0

    grad_u = sympy.Matrix([u_exact.diff(x), u_exact.diff(y)])
    flux = k * grad_u
    div_flux = flux[0].diff(x) + flux[1].diff(y)
    f = -div_flux + c * u_exact

    u_exact = sympy.lambdify([x, y], u_exact, "numpy")
    f = sympy.lambdify([x, y], f, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")

    node_coords, quad_nodes, node_groups, cell_nodes, quad_areas = load_data(quadrangle_mesh_name)
    cell_areas = utility_mvd.calculate_cell_areas(cell_nodes, node_coords)

    vector_values = calculate_vector_values(quad_nodes, node_coords, k)

    A_D_inner, f_D_inner = assemble_matrix_and_vector_for_inner_nodes(
        range(node_groups[0]), cell_nodes[:node_groups[0]], quad_nodes, vector_values, node_coords, cell_areas[:node_groups[0]], c, f)
    A_D_boundary, f_D_boundary = assemble_matrix_and_vector_for_boundary_nodes(range(node_groups[0], node_groups[1]), node_coords, u_exact)

    A_V_inner, f_V_inner = assemble_matrix_and_vector_for_inner_nodes(
        range(node_groups[1], node_groups[2]), cell_nodes[node_groups[1]:node_groups[2]], quad_nodes, vector_values, node_coords, cell_areas[node_groups[1]:node_groups[2]], c, f, Voronoi=True)
    A_V_boundary, f_V_boundary = assemble_matrix_and_vector_for_boundary_nodes(range(node_groups[2], node_groups[3]), node_coords, u_exact)

    A_D = np.concatenate((A_D_inner, A_D_boundary))
    f_D = np.concatenate((f_D_inner, f_D_boundary))

    A_V = np.concatenate((A_V_inner, A_V_boundary))
    f_V = np.concatenate((f_V_inner, f_V_boundary))

    A = np.concatenate((A_D, A_V))
    f = np.concatenate((f_D, f_V))
    u = np.linalg.solve(A, f)
    u_e = u_exact(node_coords[:, 0], node_coords[:, 1])

    L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2 = utility_mvd.calculate_errornorms(u, u_e, cell_areas, node_groups)

    if plot:
        utility_mvd.plot_results(u, u_e, node_coords, node_groups)

    #np.savetxt('test.txt', A, fmt='%+0.3f')
    
    return node_coords.shape[0], L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2


if __name__ == '__main__':
    k = np.array(
        ((1, 0),
         (0, 1))
    )
    res = calculate('meshes/rectangle/rectangle_0_quadrangle', k, plot=True)
    print(res)
