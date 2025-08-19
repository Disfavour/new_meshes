import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.tri as tri


# 2D
def get_intersection_point_of_lines(a1, a2, b1, b2):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    p_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    p_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return p_x, p_y


# Формула площади Гаусса (многоугольника)
def polygon_area(points):
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def calculate_cell_areas(cell_nodes, node_coords):
    cell_areas = []
    for current_cell_nodes in cell_nodes:
        cell_areas.append(polygon_area([node_coords[node] for node in current_cell_nodes]))
    return np.array(cell_areas)


# Надо не забыть, что без граничных узлов Вороного
def dot(y, v, cell_areas):
    return (y * v * cell_areas).sum()


def norm(y, cell_areas):
    return np.sqrt(dot(y, y, cell_areas))


def shared_dot(y, v, cell_areas, node_groups):
    y_D = np.concatenate((y[:node_groups[0]], y[node_groups[1]:node_groups[2]]))
    y_V = y[node_groups[0]:node_groups[1]]
    v_D = np.concatenate((v[:node_groups[0]], v[node_groups[1]:node_groups[2]]))
    v_V = v[node_groups[0]:node_groups[1]]
    cell_areas_D = np.concatenate((cell_areas[:node_groups[0]], cell_areas[node_groups[1]:node_groups[2]]))
    cell_areas_V = cell_areas[node_groups[0]:node_groups[1]]
    return dot(y_D, v_D, cell_areas_D) + dot(y_V, v_V, cell_areas_V)


def shared_norm(*args):
    if len(args) == 2:
        norm_D, norm_V = args
        return np.sqrt(norm_D ** 2 + norm_V ** 2)
    elif len(args) == 3:
        y, cell_areas, node_groups = args
        return np.sqrt(shared_dot(y, y, cell_areas, node_groups))


def calculate_errornorms(u, u_e, cell_areas, node_groups):
    error = u - u_e
    error_D = np.concatenate((error[:node_groups[0]], error[node_groups[1]:node_groups[2]]))
    error_V_inner = error[node_groups[0]:node_groups[1]]
    error_V = np.concatenate((error_V_inner, error[node_groups[2]:]))

    cell_areas_D = np.concatenate((cell_areas[:node_groups[0]], cell_areas[node_groups[1]:node_groups[2]]))
    cell_areas_V_inner = cell_areas[node_groups[0]:node_groups[1]]

    L_max_D = np.abs(error_D).max()
    L_max_V = np.abs(error_V).max()
    L_max = max(L_max_D, L_max_V)

    L_2_D = norm(error_D, cell_areas_D)
    L_2_V = norm(error_V_inner, cell_areas_V_inner)
    L_2 = shared_norm(L_2_D, L_2_V)
    assert np.allclose(L_2, shared_norm(error, cell_areas, node_groups))

    return L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2


def vector_dot(v, w, quad_areas):
    return ((v[:, 0] * w[:, 0] + v[:, 1] * w[:, 1]) * quad_areas).sum()
    #return (v[0] * w[0] + v[1] * w[1]) * quad_areas


def vector_norm(v, quad_areas):
    return np.sqrt(vector_dot(v, v, quad_areas))


def calculate_vector_errornorm(quadrangle_nodes, quad_areas, node_coords, grad, u):
    B = np.column_stack((
        (1, 0),
        (0, 1)
    ))

    grad_numeric = []
    grad_analytic = []
    for quad_nodes, area in zip(quadrangle_nodes, quad_areas):
        intersection_point = get_intersection_point_of_lines(*node_coords[quad_nodes[::2]], *node_coords[quad_nodes[1::2]])
        grad_i = grad(*intersection_point)
        
        e_D = node_coords[quad_nodes[2]] - node_coords[quad_nodes[0]]
        e_D_lenght = np.linalg.norm(e_D)

        e_V = node_coords[quad_nodes[3]] - node_coords[quad_nodes[1]]
        e_V_lenght = np.linalg.norm(e_V)
        
        B_new_basis = np.column_stack((
            e_D / e_D_lenght,
            e_V / e_V_lenght
        ))

        C = np.linalg.inv(B) @ B_new_basis
        grad_new_basis = np.linalg.inv(C) @ grad_i

        grad_analytic.append(grad_new_basis.reshape(2))

        grad_numeric.append(np.array(((u[quad_nodes[2]] - u[quad_nodes[0]]) / e_D_lenght, (u[quad_nodes[3]] - u[quad_nodes[1]]) / e_V_lenght)))

    grad_numeric = np.array(grad_numeric)
    grad_analytic = np.array(grad_analytic)

    error = grad_numeric - grad_analytic
    
    return vector_norm(error, quad_areas)
        


def plot_results(u_numeric, u_exact, node_coords, node_groups):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True, sharex=True, sharey=True)

    vmin = min(u_numeric.min(), u_exact.min())
    vmax = max(u_numeric.max(), u_exact.max())

    axes[0][0].tripcolor(node_coords[:node_groups[1], 0], node_coords[:node_groups[1], 1], u_numeric[:node_groups[1]], shading='gouraud', vmin=vmin, vmax=vmax)
    axes[0][0].set_title("Численное в узлах Делоне")
    axes[0][0].axis('scaled')

    axes[0][1].tripcolor(node_coords[node_groups[1]:, 0], node_coords[node_groups[1]:, 1], u_numeric[node_groups[1]:], shading='gouraud', vmin=vmin, vmax=vmax)
    axes[0][1].set_title("Численное в узлах Вороного")
    axes[0][1].axis('scaled')

    axes[1][0].tripcolor(node_coords[:, 0], node_coords[:, 1], u_numeric, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[1][0].set_title("Численное во всех узлах")
    axes[1][0].axis('scaled')

    plot = axes[1][1].tripcolor(node_coords[:, 0], node_coords[:, 1], u_exact, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[1][1].set_title("Аналитическое")
    axes[1][1].axis('scaled')

    fig.colorbar(plot, ax=axes, orientation='vertical', label='u')

    plt.show()