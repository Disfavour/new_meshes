import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.tri as tri


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
    return dot(y[:node_groups[1]], v[:node_groups[1]], cell_areas[:node_groups[1]]) + dot(y[node_groups[1]:node_groups[2]], v[node_groups[1]:node_groups[2]], cell_areas[node_groups[1]:node_groups[2]])


def shared_norm(*args):
    if len(args) == 2:
        norm_D, norm_V = args
        return np.sqrt(norm_D ** 2 + norm_V ** 2)
    elif len(args) == 3:
        y, cell_areas, node_groups = args
        return np.sqrt(shared_dot(y, y, cell_areas, node_groups))


def vector_dot(v, w, quad_areas):
    return (v[:, 0] * w[:, 0] + v[:, 1] * w[:, 1]) * quad_areas


def vector_norm(v, quad_areas):
    return np.sqrt(vector_dot(v, v, quad_areas))


def calculate_errornorms(u, u_e, cell_areas, node_groups):
    error = u - u_e

    L_max_D = np.abs(error)[:node_groups[1]].max()
    L_max_V = np.abs(error)[node_groups[1]:].max()
    L_max = max(L_max_D, L_max_V)

    L_2_D = norm(error[:node_groups[1]], cell_areas[:node_groups[1]])
    L_2_V = norm(error[node_groups[1]:node_groups[2]], cell_areas[node_groups[1]:node_groups[2]])
    L_2 = shared_norm(L_2_D, L_2_V)
    assert np.allclose(L_2, shared_norm(error, cell_areas, node_groups))

    return L_max_D, L_max_V, L_max, L_2_D, L_2_V, L_2


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