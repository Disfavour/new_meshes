import gmsh
import numpy as np
import sympy
import sys
sys.path.append('mesh_generation')
import utility


# div (k * grad u) + c*u = f
def calculate(quadrangle_mesh_name, k):
    x, y = sympy.symbols('x y')

    u = x**2 + x*y #sympy.exp(x*y)
    c = 0

    grad_u = sympy.Matrix([u.diff(x), u.diff(y)])
    flux = k * grad_u
    div_flux = flux[0].diff(x) + flux[1].diff(y)
    f = -div_flux + c * u

    u = sympy.lambdify([x, y], u, "numpy")
    f = sympy.lambdify([x, y], f, "numpy")
    c = sympy.lambdify([x, y], c, "numpy")

    gmsh.initialize()
    gmsh.open(f'{quadrangle_mesh_name}.msh')

    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Quadrangle", 1))
    element_node_tags = element_node_tags.reshape(-1, 4)

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords = node_coords.reshape(-1, 3)
    one = np.uint64(1)

    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        indices = np.argsort(node_tags)
        node_tags = node_tags[indices]
        node_coords = node_coords[indices]
    assert np.all(node_tags[:-1] < node_tags[1:])

    #quad_areas = gmsh.model.mesh.get_element_qualities(element_tags, 'volume')

    gmsh.model.mesh.split_quadrangles()

    triangle_tags, triangle_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    triangle_node_tags = triangle_node_tags.reshape(-1, 3)
    gmsh.finalize()

    loaded = np.load(f'{quadrangle_mesh_name}.npz', allow_pickle=True)
    node_groups = loaded['node_groups']
    cells = loaded['cells']

    row = np.zeros(node_tags.size)

    # Вычисляем векторные величины в центрах четырехугольных ячеек
    eb1 = np.array((1, 0))
    eb2 = np.array((0, 1))
    B = np.column_stack((eb1, eb2))

    vectors = []
    for quadrangle, quad_nodes in zip(element_tags, element_node_tags):
        e_D = (node_coords[quad_nodes[2] - one] - node_coords[quad_nodes[0] - one])[:2]
        e1 = e_D / np.linalg.norm(e_D)

        e_V = (node_coords[quad_nodes[3] - one] - node_coords[quad_nodes[1] - one])[:2]
        e2 = e_V / np.linalg.norm(e_V)
        
        B_new = np.column_stack((e1, e2))

        C = np.linalg.inv(B) @ B_new

        grad = [
            np.copy(row),
            np.copy(row)
        ]

        grad[0][quad_nodes[2] - one] = 1
        grad[0][quad_nodes[0] - one] = -1
        grad[0] /= np.linalg.norm(e_D)

        grad[1][quad_nodes[3] - one] = 1
        grad[1][quad_nodes[1] - one] = -1
        grad[1] /= np.linalg.norm(e_V)

        k_new = np.linalg.inv(C) @ k @ C

        product = [
            np.copy(row),
            np.copy(row)
        ]

        product[0] = k_new[0][0] * grad[0] + k_new[0][1] * grad[1]
        product[1] = k_new[1][0] * grad[0] + k_new[1][1] * grad[1]

        vectors.append(product)

    check_cells = []
    check_nodes = []
    # Матрица для вершин Делоне
    rows_D = []
    column_D = []
    for node_D_inner, cell_nodes in zip(range(1, node_groups[0] + one), cells):
        check_cells.append(cell_nodes)
        check_nodes.append(node_D_inner)
        div_sum = 0
        cell_boundary_lenght = 0
        for n1, n2 in zip(cell_nodes, np.roll(cell_nodes, -1)):
            rows_with_n1, _ = (element_node_tags == n1).nonzero()
            rows_with_n2, _ = (element_node_tags == n2).nonzero()
            row_with_both = np.intersect1d(rows_with_n1, rows_with_n2, assume_unique=True)[0]
            quad_nodes = element_node_tags[row_with_both]

            vn = vectors[row_with_both][0]
            if node_D_inner != quad_nodes[0]:
                vn = -vn
            
            edge_lenght = np.linalg.norm((node_coords[quad_nodes[2] - one] - node_coords[quad_nodes[0] - one])[:2])
            div_sum += vn * edge_lenght
            cell_boundary_lenght += edge_lenght


        area = utility.polygon_area(np.array([node_coords[node - one][:2] for node in cell_nodes]))
        div = div_sum# / area

        c_row = row.copy()
        c_row[node_D_inner - 1] = c(*node_coords[node_D_inner - 1][:2])
        
        rows_D.append(-div + c_row)
        #column_D.append(f(*node_coords[node_D_inner - 1][:2]))
        column_D.append(f(*node_coords[node_D_inner - 1][:2]) * area)
    
    for node_D_boundary, cell_nodes in zip(range(node_groups[0] + one, node_groups[1] + one), cells[node_groups[0]:]):
        check_cells.append(cell_nodes)
        check_nodes.append(node_D_boundary)
        new_row = row.copy()
        new_row[node_D_boundary - 1] = 1
        rows_D.append(new_row)
        column_D.append(u(*node_coords[node_D_boundary - 1][:2]))
    
    A_D = np.array(rows_D)
    b_D = np.array(column_D)

    # Матрица для вершин Вороного
    rows_V = []
    column_V = []
    for node_V_inner, cell_nodes in zip(range(node_groups[1] + one, node_groups[2] + one), cells[node_groups[1]:]):
        check_cells.append(cell_nodes)
        check_nodes.append(node_V_inner)
        div_sum = 0
        cell_boundary_lenght = 0
        for n1, n2 in zip(cell_nodes, np.roll(cell_nodes, -1)):
            rows_with_n1, _ = (element_node_tags == n1).nonzero()
            rows_with_n2, _ = (element_node_tags == n2).nonzero()
            row_with_both = np.intersect1d(rows_with_n1, rows_with_n2, assume_unique=True)[0]
            quad_nodes = element_node_tags[row_with_both]

            vn = vectors[row_with_both][1]
            if node_V_inner != quad_nodes[1]:
                vn = -vn
            
            edge_lenght = np.linalg.norm((node_coords[quad_nodes[3] - one] - node_coords[quad_nodes[1] - one])[:2])
            div_sum += vn * edge_lenght
            cell_boundary_lenght += edge_lenght

        area = utility.polygon_area(np.array([node_coords[node - one][:2] for node in cell_nodes]))
        div = div_sum# / area

        c_row = row.copy()
        c_row[node_V_inner - 1] = c(*node_coords[node_V_inner - 1][:2])
        
        rows_V.append(-div + c_row)
        #column_V.append(f(*node_coords[node_V_inner - 1][:2]))
        column_V.append(f(*node_coords[node_V_inner - 1][:2]) * area)
    
    for node_V_boundary, cell_nodes in zip(range(node_groups[2] + one, node_groups[3] + one), cells[node_groups[2]:]):
        check_cells.append(cell_nodes)
        check_nodes.append(node_V_boundary)
        new_row = row.copy()
        new_row[node_V_boundary - 1] = 1
        rows_V.append(new_row)
        column_V.append(u(*node_coords[node_V_boundary - 1][:2]))

    # for i, j in zip(check_cells, cells):
    #     print(np.array_equal(i, j), i, j)
    # print(len(check_cells), len(cells))

    # for i, j in zip(check_nodes, node_tags):
    #     print(np.array_equal(i, j), i, j)
    # print(node_groups)
    # print(len(check_nodes), len(node_tags))

    # exit()
    
    A_V = np.array(rows_V)
    b_V = np.array(column_V)

    A = np.concatenate((A_D, A_V))
    b = np.concatenate((b_D, b_V))

    x = np.linalg.solve(A, b)
    u_ex = np.array([u(*coords[:2]) for coords in node_coords])

    error = x - u_ex

    L_max_D = np.abs(error)[:node_groups[1]].max()
    L_max_V = np.abs(error)[node_groups[1]:].max()
    L_max = max(L_max_D, L_max_V)

    L_D_squared = 0
    area_D = 0
    for cell_nodes, e in zip(cells[:node_groups[1]], error):
        cell_area = utility.polygon_area([node_coords[node - one][:2] for node in cell_nodes])
        area_D += cell_area
        L_D_squared += e**2 * cell_area
    L_D = np.sqrt(L_D_squared)

    L_V_squared = 0
    area_V = 0
    for cell_nodes, e in zip(cells[node_groups[1]:node_groups[2]], error[node_groups[1]:node_groups[2]]):
        cell_area = utility.polygon_area([node_coords[node - one][:2] for node in cell_nodes])
        area_V += cell_area
        L_V_squared += e**2 * cell_area
    L_V = np.sqrt(L_V_squared)

    L = np.sqrt(L_D_squared + L_V_squared)

    assert np.allclose(area_D, area_V)


    #plot_results(x, u_ex, L_infinity, triangle_node_tags, node_coords)

    # print(x[:node_groups[0]])
    # print(u_ex[:node_groups[0]])
    # print()

    # print(x[node_groups[0]:node_groups[1]])
    # print(u_ex[node_groups[0]:node_groups[1]])
    # print()

    # print(x[node_groups[1]:node_groups[2]])
    # print(u_ex[node_groups[1]:node_groups[2]])
    # print()

    # print(x[node_groups[2]:node_groups[3]])
    # print(u_ex[node_groups[2]:node_groups[3]])
    # print()

    np.savetxt('test.txt', A_D, fmt='%+0.3f')


    # quad -> e_d, e_v
    # quad -> k * grad u
    # for quad, nodes in zip(quad_tags, quad_nodes):
    #     pass
    # for node, cell, quad

    return L_max_D, L_max_V, L_max, L_D, L_V, L




if __name__ == '__main__':
    k = np.array(
        ((1, 0),
         (0, 1))
    )
    calculate('meshes/rectangle/rectangle_1_quadrangle', k)
