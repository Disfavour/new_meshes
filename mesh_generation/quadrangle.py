import gmsh
import numpy as np
import utility


def generate(triangle_mesh, fname=None, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}

    # проблема с типо в add_circumcenter (int: numpy)
    # барицентр и центроиды
    # >1 surface вряд ли будет работать

    D_inner_nodes = []
    V_inner_nodes = []
    D_boundary_nodes = []
    V_boundary_nodes = []

    D_inner_cells = []
    V_inner_cells = []
    D_boundary_cells = []
    V_boundary_cells = []

    D_to_V = {}
    D_to_V_boundary = {}    # тут надо потом соединять с обращенным словарем
    V_to_D = {}
    V_to_D_boundary = {}    # тут реально словарь для граничных
    
    # граничные ячейки вороного замкнуть

    for dim, surf_tag in gmsh.model.get_entities(2):
        boundaries = gmsh.model.get_boundary(((dim, surf_tag),), oriented=False)

        D_inner_nodes = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=False)[0]
        D_boundary_nodes = np.setdiff1d(gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=True)[0], D_inner_nodes)
    
        triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
        triangle_nodes = triangle_nodes.reshape(-1, 3)
        triangles_to_nodes = {triangle: nodes for triangle, nodes in zip(triangle_tags, triangle_nodes)}
        #assert np.all(triangle_tags[:-1] < triangle_tags[1:])
        #print(triangle_tags, triangle_tags.size, triangle_tags.max())
        #assert triangle_tags.size == triangle_tags.max()

        center_tags = utility.add_triangle_centers(dim, surf_tag, triangle_nodes)
        triangle_to_center = {triangle: center for triangle, center in zip(triangle_tags, center_tags)}

        V_inner_nodes.extend(center_tags)
        V_to_D.update(zip(center_tags, triangle_nodes))

        # после добавления центров обновляем
        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
        assert np.all(node_tags[:-1] < node_tags[1:])
        assert node_tags.size == node_tags.max()
        node_coords = node_coords.reshape(-1, 3)
        one = np.uint64(1)

        edge_nodes = gmsh.model.mesh.get_element_edge_nodes(triangle_type, surf_tag)
        edge_tags, edge_orientations = gmsh.model.mesh.get_edges(edge_nodes)
        triangle_to_edges = {triangle: edges for triangle, edges in zip(triangle_tags, edge_tags.reshape(-1, 3))}
        edge_to_triangles = utility.reverse_dict(triangle_to_edges)
        
        max_node_tag = np.uint64(gmsh.model.mesh.get_max_node_tag())
        quadrangle_nodes = []
        boundary_to_nodes_and_coords = {boundary: ([], []) for boundary in boundaries}
        for edge, linked_triangles in edge_to_triangles.items():
            nodes = np.sort(edge_to_nodes[edge])

            # В направлении вершины Делоне с большим индексом
            vector_D = np.array(node_coords[nodes[1] - one] - node_coords[nodes[0] - one])[:2]
            vector_V = np.array((-vector_D[1], vector_D[0]))

            quad_nodes = None
            quad_node_coords = None

            if len(linked_triangles) == 2:
                V_nodes = (triangle_to_center[linked_triangles[0]], triangle_to_center[linked_triangles[1]])
                vector = np.array(node_coords[V_nodes[1] - one] - node_coords[V_nodes[0] - one])[:2]
                if np.dot(vector, vector_V) < 0:
                    V_nodes = V_nodes[::-1]

                # (nodes[0], nodes[1]) (V_nodes[0], V_nodes[1]) 90 градусов против часовой
                quad_nodes = [nodes[0], V_nodes[0], nodes[1], V_nodes[1]]
                quad_node_coords = [node_coords[node - one] for node in quad_nodes]

            else:
                coords1, parametric_coords1, dim1, tag1 = gmsh.model.mesh.get_node(nodes[0])
                coords2, parametric_coords2, dim2, tag2 = gmsh.model.mesh.get_node(nodes[1])
                boundary = utility.get_boundary(dim1, tag1, dim2, tag2)

                edge_center = (coords1 + coords2) / 2
                
                max_node_tag += one
                boundary_to_nodes_and_coords[boundary][0].append(max_node_tag)
                boundary_to_nodes_and_coords[boundary][1].extend(edge_center)

                V_nodes = (triangle_to_center[linked_triangles[0]], max_node_tag)
                vector = np.array(edge_center - node_coords[V_nodes[0] - one])[:2]
                if np.dot(vector, vector_V) < 0:
                    V_nodes = V_nodes[::-1]
                    quad_nodes = [nodes[0], V_nodes[0], nodes[1], V_nodes[1]]
                    quad_node_coords = [node_coords[quad_nodes[0] - one]] + [edge_center] + [node_coords[node - one] for node in quad_nodes[2:]]
                else:
                    quad_nodes = [nodes[0], V_nodes[0], nodes[1], V_nodes[1]]
                    quad_node_coords = [node_coords[node - one] for node in quad_nodes[:-1]] + [edge_center]

                V_boundary_nodes.append(max_node_tag)

            assert utility.is_counter_clockwise(quad_node_coords)
            if not utility.is_counter_clockwise(quad_node_coords):
                quad_nodes.reverse()

            quadrangle_nodes.extend(quad_nodes)

            if len(linked_triangles) == 1:
                # Словари для граничных ячеек неполные, тк содержат только граничные ячейки (а не полную ячейку)
                # для граничной вершины Вороного ячейка - треугольник в который она входит
                V_to_D_boundary.update(((max_node_tag, triangles_to_nodes[linked_triangles[0]]),))

                for node in nodes:
                    if node not in D_to_V_boundary:
                        D_to_V_boundary[node] = []
                    D_to_V_boundary[node].append(max_node_tag)

        for boundary in boundaries:
            gmsh.model.mesh.add_nodes(*boundary, boundary_to_nodes_and_coords[boundary][0], boundary_to_nodes_and_coords[boundary][1])

        gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)
        #gmsh.model.mesh.add_elements_by_type(surf_tag, quadrangle_type, [], quadrangle_nodes) range(1, len(quadrangle_nodes) // 4 + 1)
        # gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)
        gmsh.model.mesh.add_elements_by_type(surf_tag, quadrangle_type, [], quadrangle_nodes)

    # при реверсе порядок против часовой не сохраняется
    D_to_V = utility.reverse_dict(V_to_D)

    V_to_D.update(V_to_D_boundary)

    # Граничная вершина Делоне может не войти в свою ячейку (например на углах), поэтому добавляем ее
    for k in D_to_V_boundary:
        D_to_V_boundary[k].append(k)

    # добавляем тоже в случайном порядке
    for k, vs in D_to_V_boundary.items():
        for v in vs:
            D_to_V[k].append(v)
    
    # D_to_V содержит ячейки для всех вершин Делоне, но вершины ячеек в рандомном порядке (надо сортировать, чтобы была выпуклая)
    # Были добавлены вершины Вороного на границах -> надо обновить
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    indices = np.argsort(node_tags)
    #assert np.all(node_tags[:-1] < node_tags[1:])
    assert node_tags.size == node_tags.max()
    node_coords = node_coords.reshape(-1, 3)
    node_tags = node_tags[indices]
    node_coords = node_coords[indices]
    one = np.uint64(1)
    
    for k, vs in D_to_V.items():
        D_to_V[k] = utility.sort_counter_clockwise(vs, [node_coords[node - one] for node in vs])

    #different_nodes = (D_inner_nodes, D_boundary_nodes, V_inner_nodes, V_boundary_nodes)    # old
    different_nodes = (D_inner_nodes, V_inner_nodes, D_boundary_nodes, V_boundary_nodes)
    all_nodes = np.concatenate(different_nodes)
    lenghts = [len(arr) for arr in different_nodes]
    node_groups = np.array([sum(lenghts[:i]) for i in range(1, len(lenghts) + 1)], dtype=np.uint64)
    
    # (
    #     D_inner_nodes.size,
    #     D_inner_nodes.size + len(V_inner_nodes),
    #     D_inner_nodes.size + len(V_inner_nodes) + D_boundary_nodes.size,
    #     all_nodes.size)

    new_tags = range(1, all_nodes.size + 1)
    gmsh.model.mesh.renumber_nodes(all_nodes, new_tags)
    #gmsh.model.mesh.renumber_nodes()

    old_to_new = {old: new for old, new in zip(all_nodes, new_tags)}

    new_D_to_V = {}
    new_V_to_D = {}

    for old_d, new_d in zip((D_to_V, V_to_D), (new_D_to_V, new_V_to_D)):
        for k, vs in old_d.items():
            new_k = old_to_new[k]
            new_d[new_k] = []
            for v in vs:
                new_d[new_k].append(old_to_new[v])
    
    node_to_cell = {}
    node_to_cell.update(new_D_to_V)
    node_to_cell.update(new_V_to_D)

    cells = []

    for node in new_tags:
        cells.append(np.array(node_to_cell[node], dtype=np.uint64))

    cells = np.array(cells, dtype=object)

    # element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()
    # element_type = triangle_type if triangle_type in element_types else quadrangle_type
    # element_tags = element_tags[np.argwhere(element_types == element_type)[0][0]]
    # assert np.all(element_tags[:-1] < element_tags[1:])
    # assert element_tags.size == element_tags.max()

    gmsh.model.mesh.renumber_elements()

    # Не перенумеровываем вершины, тк сделали свой порядок, чтобы матрица выглядела хорошо

    # node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    # assert np.all(node_tags[:-1] < node_tags[1:])
    # assert node_tags.size == node_tags.max()
    
    #gmsh.model.mesh.renumber_nodes()

    assert gmsh.model.mesh.get_duplicate_nodes().size == 0

    #gmsh.model.mesh.remove_duplicate_nodes()

    assert utility.is_all_counter_clockwise()

    if fname is not None:
        gmsh.write(f'{fname}.msh')
        np.savez_compressed(f'{fname}.npz', node_groups=node_groups, cells=cells)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate('meshes/rectangle/rectangle_0_triangle.msh', 'meshes/rectangle/rectangle_0_quadrangle', ui=True)
