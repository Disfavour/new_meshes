import gmsh
import utility
import numpy as np


def simultaneous_version(niter):
    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

    for iter in range(niter):
        for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
            surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]

            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
            triangle_to_nodes = {triangle: nodes for triangle, nodes in zip(triangle_tags, triangle_nodes.reshape(-1, 3))}
            node_to_triangles = utility.reverse_dict(triangle_to_nodes)

            # get_barycenters привязан к тегу, поэтому нельзя вынести из цикла
            triangle_barycenters = gmsh.model.mesh.get_barycenters(triangle_type, surf_tag, fast=False, primary=True).reshape(-1, 3)
            triangle_areas = gmsh.model.mesh.get_element_qualities(triangle_tags, qualityName='volume')

            triangle_to_barycenter = {triangle: barycenter for triangle, barycenter in zip(triangle_tags, triangle_barycenters)}
            triangle_to_area = {triangle: area for triangle, area in zip(triangle_tags, triangle_areas)}

            node_tags_inner, node_coords_inner, _ = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=False, returnParametricCoord=False)

            for inner_node in node_tags_inner:
                areas = np.array([triangle_to_area[triangle] for triangle in node_to_triangles[inner_node]])
                barycenters = np.array([triangle_to_barycenter[triangle] for triangle in node_to_triangles[inner_node]])

                barycenter_area_product = barycenters * areas.reshape(-1, 1)
                new_coord = barycenter_area_product.sum(axis=0) / areas.sum()

                gmsh.model.mesh.set_node(inner_node, new_coord, [])


def sequential_version(niter):
    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    one = np.uint64(1)

    for iter in range(niter):
        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
        node_coords = node_coords.reshape(-1, 3)

        for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
            surf_tag = gmsh.model.get_entities_for_physical_group(dim, physical_tag)[0]

            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
            triangle_to_nodes = {triangle: nodes for triangle, nodes in zip(triangle_tags, triangle_nodes.reshape(-1, 3))}
            node_to_triangles = utility.reverse_dict(triangle_to_nodes)

            node_tags_inner, node_coords_inner, _ = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=False, returnParametricCoord=False)

            for inner_node in node_tags_inner:
                triangle_areas = gmsh.model.mesh.get_element_qualities(node_to_triangles[inner_node], qualityName='volume')

                all_coords = [[node_coords[node - one] for node in triangle_to_nodes[triangle]] for triangle in node_to_triangles[inner_node]]

                barycenters = np.sum(all_coords, axis=1) / 3
                barycenter_area_product = barycenters * triangle_areas.reshape(-1, 1)
                new_coord = barycenter_area_product.sum(axis=0) / triangle_areas.sum()

                gmsh.model.mesh.set_node(inner_node, new_coord, [])

                node_coords[inner_node - one] = new_coord


if __name__ == '__main__':
    import time

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open('meshes/msh/rectangle_6_triangle.msh')

    start_time = time.time()
    sequential_version(1)
    print(f'Elapsed time: {time.time() - start_time:.2f}')

    gmsh.finalize()
