import gmsh
import utility
import numpy as np


def simultaneous_version(niter):
    one = np.uint64(1)

    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}
    node_to_edges = utility.reverse_dict(edge_to_nodes)
    node_to_adjacent_nodes = {}
    for inner_node, edges in node_to_edges.items():
        adjacent_nodes = []
        for edge in edges:
            for edge_node in edge_to_nodes[edge]:
                if edge_node != inner_node:
                    adjacent_nodes.append(edge_node)
        node_to_adjacent_nodes[inner_node] = adjacent_nodes
    
    for iter in range(niter):
        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
        node_coords = node_coords.reshape(-1, 3)    

        for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
            surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]

            node_tags_inner, node_coords_inner, _ = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=False, returnParametricCoord=False)

            for inner_node in node_tags_inner:
                adjacent_nodes_coords = np.array([node_coords[adjacent_node - one] for adjacent_node in node_to_adjacent_nodes[inner_node]])
                new_coord = adjacent_nodes_coords.sum(axis=0) / adjacent_nodes_coords.shape[0]

                gmsh.model.mesh.set_node(inner_node, new_coord, [])


def sequential_version(niter):
    one = np.uint64(1)
    
    gmsh.model.mesh.createEdges()
    edge_tags, edge_nodes = gmsh.model.mesh.get_all_edges()
    edge_to_nodes = {edge: nodes for edge, nodes in zip(edge_tags, edge_nodes.reshape(-1, 2))}
    node_to_edges = utility.reverse_dict(edge_to_nodes)
    node_to_adjacent_nodes = {}
    for inner_node, edges in node_to_edges.items():
        adjacent_nodes = []
        for edge in edges:
            for edge_node in edge_to_nodes[edge]:
                if edge_node != inner_node:
                    adjacent_nodes.append(edge_node)
        node_to_adjacent_nodes[inner_node] = adjacent_nodes
    
    for iter in range(niter):
        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
        node_coords = node_coords.reshape(-1, 3)

        for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
            surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]

            node_tags_inner, node_coords_inner, _ = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=False, returnParametricCoord=False)

            for inner_node in node_tags_inner:
                adjacent_nodes_coords = np.array([node_coords[adjacent_node - one] for adjacent_node in node_to_adjacent_nodes[inner_node]])
                new_coord = adjacent_nodes_coords.sum(axis=0) / adjacent_nodes_coords.shape[0]

                gmsh.model.mesh.set_node(inner_node, new_coord, [])
                node_coords[inner_node - one] = new_coord


if __name__ == '__main__':
    import time

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open('meshes/msh/rectangle_6_triangle.msh')

    start_time = time.time()
    simultaneous_version(1)
    print(f'Elapsed time: {time.time() - start_time:.2f}')

    gmsh.finalize()
