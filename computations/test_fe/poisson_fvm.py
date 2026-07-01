import gmsh
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import sympy
import matplotlib.pyplot as plt


def read_mesh_from_file(mesh_fname):
    gmsh.initialize()
    gmsh.open(mesh_fname)

    node_tags, nodes, _ = gmsh.model.mesh.get_nodes()
    cell_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    cell_tags, cells = gmsh.model.mesh.get_elements_by_type(cell_type)

    cell_edge_nodes = gmsh.model.mesh.get_element_edge_nodes(cell_type)
    gmsh.model.mesh.create_edges()
    cell_edges, edge_orientations = gmsh.model.mesh.get_edges(cell_edge_nodes)
    edges, edge_nodes = gmsh.model.mesh.getAllEdges()
    #gmsh.fltk.run()
    #print(gmsh.model.mesh.get_element_qualities(cell_tags, 'volume'))
    triangle_areas = gmsh.model.mesh.get_element_qualities(cell_tags, 'volume')
    triangle_mean_area = triangle_areas.mean()
    gmsh.finalize()

    nodes = nodes.reshape(-1, 3)[:, :2]
    assert node_tags.size == node_tags.max()
    if not np.all(node_tags[:-1] < node_tags[1:]):
        nodes = nodes[np.argsort(node_tags)]
    
    cells = cells.reshape(-1, 3) - 1    # .astype(int)

    cell_edges = cell_edges.reshape(-1, 3) - 1
    edge_nodes = edge_nodes.reshape(-1, 2) - 1
    edges -= 1

    if not np.all(edges[:-1] < edges[1:]):
        sorted_indexes = np.argsort(edges)
        edges = edges[sorted_indexes]
        edge_nodes = edge_nodes[sorted_indexes]
    
    unique, counts = np.unique(cell_edges, return_counts=True)
    boundary_edges = unique[counts==1]
    unique[:boundary_edges.size] = unique[-boundary_edges.size:]
    unique[-boundary_edges.size:] = boundary_edges

    edge_nodes = edge_nodes[unique]

    g = edges.size - boundary_edges.size
    # мы изменили номера ребер, а тут нет cell_edges

    mask1 = cell_edges < boundary_edges.size
    mask2 = cell_edges >= g

    cell_edges[mask1] += g
    cell_edges[mask2] -= g

    return nodes, cells, cell_edges, edges, edge_nodes, g, triangle_areas


def compute_polygon_areas(x, y):
    '''Формула площади Гаусса (многоугольника)'''
    return np.abs(np.sum(x * np.roll(y, -1, axis=0), axis=0) - np.sum(y * np.roll(x, -1, axis=0), axis=0)) / 2


def centroid(A, B, C):
    return (A + B + C).T / 3


def circumcenter(A, B, C):
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D
    return np.array((Ux, Uy)).T


def solve(mesh_fname, f, ue, center, K):
    nodes, cells, cell_edges, edges, edge_nodes, g, triangle_areas = read_mesh_from_file(mesh_fname)
    cell_coords = nodes[cells]

    edge_node_coords = nodes[edge_nodes]
    edge_center_coords = edge_node_coords.sum(axis=1) / 2

    # # середины ребер
    # x1, y1 = ((cell_coords[:, 1] + cell_coords[:, 0]) / 2).T
    # x2, y2 = ((cell_coords[:, 2] + cell_coords[:, 1]) / 2).T
    # x3, y3 = ((cell_coords[:, 0] + cell_coords[:, 2]) / 2).T

    # D = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
    # a = np.stack((y2 - y3, y3 - y1, y1 - y2), axis=-1) / D[:, np.newaxis]
    # b = np.stack((x3 - x2, x1 - x3, x2 - x1), axis=-1) / D[:, np.newaxis]
    # grad1 = np.stack((a, b), axis=1)
    # print(grad1)
    # Этот аналитический градиент совпадает с интегральным (np.allclose)

    #centroids = cell_coords.sum(axis=1) / 3

    cell_centers = center(*np.transpose(cell_coords, axes=(1, 2, 0)))

    # по каждому ребру считать интеграл и записывать в уравнения узлов

    vs = cell_coords - cell_centers[:, np.newaxis]
    lenghts = np.linalg.norm(vs, axis=2)
    vs /= lenghts[:, :, np.newaxis]
    #vs1 = vs.copy()
    n = np.flip(vs.copy(), axis=2)
    n[:, :, 1] *= -1

    # Избыточное вычисление длины ребра (не нужно вычислять для каждого треугольника)
    #print(cells, cells.shape)
    #print(np.roll(cells, -1, axis=1))
    cell_vectors = nodes[np.roll(cells, -1, axis=1)] - nodes[cells]
    edge_lenghts = np.linalg.norm(cell_vectors, axis=2)
    cell_vectors /= edge_lenghts[:, :, np.newaxis]
    #print(edge_lenghts.shape)
    #print(cell_vectors, cell_vectors.shape)
    cell_edge_normals = np.flip(cell_vectors, axis=2)
    cell_edge_normals[:, :, 1] *= -1
    #print(cell_edge_normals, cell_edge_normals.shape)


    #print(edge_lenghts)

    # не получилось сразу, разделил на компоненты. Можно еще через формулы компонентов посмотреть
    # grad = edge_lenghts[:, :, np.newaxis] * cell_edge_normals / triangle_areas[:, np.newaxis, np.newaxis]
    # print(grad.shape)

    grad_x = edge_lenghts * cell_edge_normals[:, :, 0]
    grad_y = edge_lenghts * cell_edge_normals[:, :, 1]
    grad = np.stack((grad_x, grad_y), axis=1)
    grad /= triangle_areas[:, np.newaxis, np.newaxis]

    q = -K @ grad

    integral = np.sum(q[:, np.newaxis]*n[:, :, :, np.newaxis], axis=2)*lenghts[:, :, np.newaxis]

    row = np.repeat(cell_edges, 3)
    col = np.tile(cell_edges, 3).flatten()
    data = integral.flatten()

    row2 = np.repeat(np.roll(cell_edges, 1, 1), 3)
    col2 = np.tile(cell_edges, 3).flatten()
    data2 = -integral.flatten()

    row = np.concatenate((row, row2))
    col = np.concatenate((col, col2))
    data = np.concatenate((data, data2))

    A = coo_array((data, (row, col)), shape=(edges.size, edges.size))
    A.eliminate_zeros()
    A = A.tocsr()
    b = f(*edge_center_coords[:g].T)

    #np.savetxt('A_new_approximation.txt', A.toarray(), fmt='%5.2f')

    #areas = np.bincount(cell_edges.flatten().astype(int), weights=np.repeat(compute_polygon_areas(*nodes[cells].T)/3, 3))

    volume_parts = np.concatenate((nodes[edge_nodes[cell_edges]], np.broadcast_to(cell_centers[:, np.newaxis, np.newaxis], (cells.shape[0], 3, 1, 2))), axis=2)
    areas = np.bincount(cell_edges.flatten().astype(int), weights=compute_polygon_areas(*volume_parts.T).T.flat)

    b *= areas[:g]

    # lifting
    A.resize((g, edges.size))
    u1 = np.zeros(edges.size)
    u1[g:] = ue(*edge_center_coords[g:].T)
    b -= A @ u1

    A.resize((g, g))

    # tmp = A.T - A
    # print(tmp.min(), tmp.max())

    #np.savetxt('A_new_approximation.txt', A.toarray(), fmt='%5.2f')

    u = spsolve(A, b)
    u = np.concatenate((u, u1[g:]))

    ue = ue(*edge_center_coords.T)

    Lmax = np.abs(u - ue).max()
    L2 = np.sqrt(((u - ue)**2 * areas).sum())

    return edges.size, L2, Lmax


def setup_problem(K):
    x, y = sympy.symbols('x y')

    u = sympy.exp(x*y)

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = K * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], -div, "numpy")
    u = sympy.lambdify([x, y], u, "numpy")

    return f, u


def experiments():
    data = []
    for i in range(1, 11):
        mesh_fname = f'meshes/rectangle/rectangle_{i}_triangle.msh'
        res = solve(mesh_fname, *setup_problem(), circumcenter)

        print(res)
        data.append(res)
    
    data = np.array(data)

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5), layout='constrained')

    axs[0].plot(data[:, 0], data[:, 3], '-o')
    axs[0].set_ylabel(r'$L_2$')
    axs[0].set_xlabel('edges')

    axs[1].plot(data[:, 0], data[:, 4], '-o')
    axs[1].set_ylabel(r'$L_\infty$')
    axs[1].set_xlabel('edges')

    for ax in axs.flat:
        ax.grid()
        ax.loglog()
    
    plt.savefig('new_approximation_on_triangle_mesh.pdf', transparent=True)
    plt.show()


if __name__ == '__main__':
    K = np.array((
        (1, 3),
        (3, 10)
    ))
    mesh_fname = f'meshes/rectangle/rectangle_{1}_triangle.msh'
    res = solve(mesh_fname, *setup_problem(K), centroid, K)
    print(res)
    
    #experiments()
    # mesh_fname = f'meshes/rectangle/rectangle_{1}_triangle.msh'
    # res = solve(mesh_fname, *setup_problem())
    # print(res)