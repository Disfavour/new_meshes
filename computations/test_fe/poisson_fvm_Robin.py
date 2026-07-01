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

    # gmsh.fltk.run()
    # exit()
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
    #edges = edges[unique] # их перенумеровали как бы

    g = edges.size - boundary_edges.size
    # мы изменили номера ребер, а тут нет cell_edges

    mask1 = cell_edges < boundary_edges.size
    mask2 = cell_edges >= g

    cell_edges[mask1] += g
    cell_edges[mask2] -= g

    # Надо еще и в edges изменить

    return nodes, cells, cell_edges, edges, edge_nodes, g, triangle_areas


def compute_polygon_areas(x, y):
    '''Формула площади Гаусса (многоугольника)'''
    return np.abs(np.sum(x * np.roll(y, -1, axis=0), axis=0) - np.sum(y * np.roll(x, -1, axis=0), axis=0)) / 2


def centroid(triangle_node_coords):
    return triangle_node_coords.sum(axis=1) / 3


def circumcenter(triangle_node_coords):
    Ax, Ay = triangle_node_coords[:, 0].T
    Bx, By = triangle_node_coords[:, 1].T
    Cx, Cy = triangle_node_coords[:, 2].T
    D = 2 * (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    Ux = ((Ax**2 + Ay**2)*(By-Cy) + (Bx**2 + By**2)*(Cy-Ay) + (Cx**2 + Cy**2)*(Ay-By)) / D
    Uy = ((Ax**2 + Ay**2)*(Cx-Bx) + (Bx**2 + By**2)*(Ax-Cx) + (Cx**2 + Cy**2)*(Bx-Ax)) / D
    return np.stack((Ux, Uy), axis=1)


def incenter(triangle_node_coords):
    edge_lenghts = np.linalg.norm(np.roll(triangle_node_coords, -1, axis=1) - triangle_node_coords, axis=2)
    return np.sum(triangle_node_coords * np.roll(edge_lenghts, -1, axis=1)[:, :, np.newaxis], axis=1) / edge_lenghts.sum(axis=1)[:, np.newaxis]


def orthocenter(triangle_node_coords):
    ab = triangle_node_coords[:, 1:] - triangle_node_coords[:, :2]
    # a = ab[:, 0], b = ab[:, 1]
    c = np.sum(triangle_node_coords[:, 2::-2] * ab, axis=2)
    return np.linalg.solve(ab, c)


def solve(K, r, c, f, bc, ue, center, mesh_fname):
    nodes, cells, cell_edges, edges, edge_nodes, g, triangle_areas = read_mesh_from_file(mesh_fname)
    cell_coords = nodes[cells]

    edge_node_coords = nodes[edge_nodes]
    edge_center_coords = edge_node_coords.sum(axis=1) / 2

    cell_centers = center(cell_coords)

    #areas = np.bincount(cell_edges.flatten().astype(int), weights=np.repeat(compute_polygon_areas(*nodes[cells].T)/3, 3))
    volume_parts = np.concatenate((nodes[edge_nodes[cell_edges]], np.broadcast_to(cell_centers[:, np.newaxis, np.newaxis], (cells.shape[0], 3, 1, 2))), axis=2)
    areas = np.bincount(cell_edges.flatten().astype(int), weights=compute_polygon_areas(*volume_parts.T).T.flat)

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

    #gradf = edge_lenghts[:, np.newaxis, :] @ cell_edge_normals 20, 1, 2 а надо 20, 2, 3

    q = -K @ grad

    integral = np.sum(q[:, np.newaxis]*n[:, :, :, np.newaxis], axis=2)*lenghts[:, :, np.newaxis]

    row = np.repeat(cell_edges, 3)
    col = np.tile(cell_edges, 3).flatten()
    data = integral.flatten()

    row2 = np.repeat(np.roll(cell_edges, 1, 1), 3)
    col2 = np.tile(cell_edges, 3).flatten()
    data2 = -integral.flatten()

    # ru
    row3 = edges.flatten()
    col3 = edges.flatten()
    data3 = r * areas

    # bc
    boundary_edge_nodes = nodes[edge_nodes[g:]]
    boundary_edge_vectors = boundary_edge_nodes[:, 1] - boundary_edge_nodes[:, 0]
    boundary_edge_vector_lenghts = np.linalg.norm(boundary_edge_vectors, axis=1)
    boundary_edge_vectors /= boundary_edge_vector_lenghts[:, np.newaxis]
    boundary_edge_normals = np.flip(boundary_edge_vectors, axis=1)
    boundary_edge_normals[:, 1] *= -1
    bc = bc(*edge_center_coords[g:].T, *boundary_edge_normals.T) * boundary_edge_vector_lenghts
    # bc * areas[g:]

    rowbc = edges[g:]
    colbc = edges[g:]
    databc = c * boundary_edge_vector_lenghts

    row = np.concatenate((row, row2, row3, rowbc))
    col = np.concatenate((col, col2, col3, colbc))
    data = np.concatenate((data, data2, data3, databc))

    A = coo_array((data, (row, col)), shape=(edges.size, edges.size))
    A.eliminate_zeros()
    A = A.tocsr()
    b = f(*edge_center_coords.T)

    b *= areas
    b[g:] += bc

    # tmp = A.T - A
    # print(tmp.min(), tmp.max())

    # np.savetxt('A_new_approximation.txt', A.toarray(), fmt='%5.2f')

    u = spsolve(A, b)

    ue = ue(*edge_center_coords.T)

    Lmax = np.abs(u - ue).max()
    L2 = np.sqrt(((u - ue)**2 * areas).sum())

    return edges.size, L2, Lmax


def setup_problem(K, r, c):
    # div (-k * grad u) + ru = f
    # -q*n + cu = bc
    x, y, nx, ny = sympy.symbols('x y nx ny')
    n = sympy.Matrix([nx, ny])

    u = sympy.exp(sympy.sqrt(1 + x) * y)

    grad = sympy.Matrix([u.diff(x), u.diff(y)])
    q = -K * grad
    div = q[0].diff(x) + q[1].diff(y)

    f = sympy.lambdify([x, y], div + r*u, "numpy")
    bc = sympy.lambdify([x, y, nx, ny], -q.dot(n) + c*u, "numpy")

    u = sympy.lambdify([x, y], u, "numpy")
    #q = sympy.lambdify([x, y], q, "numpy")

    return K, r, c, f, bc, u


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
    r = 1
    c = 1
    mesh_fname = f'meshes/rectangle/rectangle_{4}_triangle.msh'
    res = solve(*setup_problem(K, r, c), centroid, mesh_fname)
    print(res)
    
    #experiments()
    # mesh_fname = f'meshes/rectangle/rectangle_{1}_triangle.msh'
    # res = solve(mesh_fname, *setup_problem())
    # print(res)