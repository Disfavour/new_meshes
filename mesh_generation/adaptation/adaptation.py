import gmsh
import numpy as np


def adaptation():
    gmsh.initialize()

    dim = 2

    fname = 'test.msh'
    number_of_iterations = 100
    
    circle_tag = gmsh.model.occ.add_circle(0, 0, 0, 1)
    ellipse_tag = gmsh.model.occ.add_ellipse(0, 0, 0, 2.5, 1.5)
    circle2_tag = gmsh.model.occ.add_circle(0, 0, 0, 3)

    flower_radius = 4
    N = 100
    points = []
    for i in range(N):
        fi = 2 * np.pi * i / N
        r = flower_radius + 0.5 * np.cos(8 * fi)
        x = -r * np.cos(fi)
        y = -r * np.sin(fi)
        points.append(gmsh.model.occ.addPoint(x, y, 0))
    flower = gmsh.model.occ.addSpline(points + [points[0]])

    areas = [circle_tag, ellipse_tag, circle2_tag, flower]

    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = map(lambda x: 1.5 * x, gmsh.model.getBoundingBox(-1, -1))
    triangle_edge_lenght = min(xmax - xmin, ymax - ymin) / 100  # = 0.3

    # linear        1 - 1/distance * (np.linalg.norm(x) - inner_radius)
    # quadratic     (1/distance * (np.linalg.norm(x) - inner_radius) - 1) ** 2
    # np.exp(10*(-(np.linalg.norm(x) - inner_radius)))
    weight_functions = [lambda x: 1, lambda x: np.exp(5*(-(np.linalg.norm(x) - 1))), lambda x: 1, lambda x: np.exp(5*(-(np.linalg.norm(x) - 3)))]
    

    #
    curve_loops = [gmsh.model.occ.add_curve_loop([area]) for area in areas]
    plane_surfaces = [gmsh.model.occ.add_plane_surface([curve_loops[0]])] + [gmsh.model.occ.add_plane_surface([cl1, cl2]) for cl1, cl2 in zip(curve_loops[1:], curve_loops)]

    gmsh.model.occ.synchronize()
    
    physical_groups = [gmsh.model.add_physical_group(dim, [plane_surface]) for plane_surface in plane_surfaces]

    # nodes
    triangle_height = np.sqrt(triangle_edge_lenght ** 2 - (triangle_edge_lenght / 2) ** 2)

    x, y = np.meshgrid(np.arange(xmin, xmax, triangle_edge_lenght), np.arange(ymin, ymax, triangle_height), indexing='xy')
    x[1::2] += triangle_edge_lenght / 2

    #gmsh.model.mesh.generate(2)
    gmsh.model.mesh.add_nodes(dim, 1, [], [coord for xyz in zip(x.flat, y.flat, np.zeros(x.size)) for coord in xyz])

    triangulation = gmsh.model.mesh.triangulate([coord for xy in zip(x.flat, y.flat) for coord in xy])

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    gmsh.model.mesh.addElementsByType(1, triangle_type, [], triangulation)

    # pin boundary
    one = np.uint64(1)
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_coords = node_coords.reshape(-1, 3)

    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    triangle_nodes = triangle_nodes.reshape(-1, 3)

    node_area_num = []
    for node_coord in node_coords:
        area_num = 100
        for plane_surface in plane_surfaces:
            if gmsh.model.is_inside(dim, plane_surface, node_coord):
                area_num = plane_surface
                break
        node_area_num.append(area_num)

    #areas_nodes = [set() for i in range(len(plane_surfaces))]
    curves_nodes = [set() for i in range(len(curve_loops))]

    for tags in triangle_nodes:
        #coords = [node_coords[tag - one] for tag in tags]
        area_nums = [node_area_num[tag - one] for tag in tags]

        if len(set(area_nums)) == 1:
            continue

        groups = [[], []]
        for node_tag, area_num in zip(tags, area_nums):
            if area_num == area_nums[0]:
                groups[0].append(node_tag)
            else:
                groups[1].append(node_tag)
        
        curve = min(area_nums)    # sorted(set(area_nums))[0]

        closest_coords = [[], []]
        distances = [[], []]
        for i, group in enumerate(groups):
            for node_tag in group:
                node_coord = node_coords[node_tag - one]
                closest_coord, _ = gmsh.model.get_closest_point(1, curve, node_coord)
                closest_coords[i].append(closest_coord)
                distance = np.linalg.norm(node_coord - closest_coord)
                distances[i].append(distance)
        
        distances = [max(i) for i in distances]
        group_to_move = np.argmin(distances)
        
        for tag, new_coord in zip(groups[group_to_move], closest_coords[group_to_move]):
            gmsh.model.mesh.set_node(tag, new_coord, [])
            node_coords[tag - one] = new_coord
        
        # update area num
        for node_tag in groups[group_to_move]:
            for plane_surface in plane_surfaces:
                if gmsh.model.is_inside(dim, plane_surface, node_coord):
                    node_area_num[node_tag - one] = plane_surface
                    break
        
        curves_nodes[curve - 1].update(groups[group_to_move])
    
    areas_nodes = [set() for i in range(len(plane_surfaces))]
    for node_tag, node_coord in zip(node_tags, node_coords):
        surfaces = []
        for plane_surface in plane_surfaces:
            if gmsh.model.is_inside(dim, plane_surface, node_coord):
                areas_nodes[plane_surface - 1].add(node_tag)
                surfaces.append(plane_surface)

    triangles_in_areas = [np.all(np.isin(triangle_nodes, list(area_nodes)), axis=1).nonzero()[0] for area_nodes in areas_nodes] # row numbers

    gmsh.model.mesh.remove_elements(dim, 1)
    gmsh.model.mesh.reclassify_nodes()

    for plane_surface, nodes in zip(plane_surfaces, areas_nodes):
        nodes = list(nodes)
        gmsh.model.mesh.add_nodes(dim, plane_surface, nodes, np.array([node_coords[node_tag - one] for node_tag in nodes]).flatten())
    
    for curve, nodes in zip(curve_loops, curves_nodes):
        nodes = list(nodes)
        gmsh.model.mesh.add_nodes(dim-1, curve, nodes, np.array([node_coords[node_tag - one] for node_tag in nodes]).flatten())

    for plane_surface, triangles in zip(plane_surfaces, triangles_in_areas):
        gmsh.model.mesh.addElementsByType(plane_surface, triangle_type, [], triangle_nodes[list(triangles)].flatten())

    #gmsh.model.mesh.reclassify_nodes()
    gmsh.model.mesh.remove_duplicate_nodes()
    #gmsh.model.mesh.renumber_nodes()

    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_to_coords = {node: coords for node, coords in zip(node_tags, node_coords.reshape(-1, 3))}

    triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type)
    triangle_to_nodes = {triangle: nodes for triangle, nodes in zip(triangle_tags, triangle_nodes.reshape(-1, 3))}
    node_to_triangles = {}
    for k, vs in triangle_to_nodes.items():
        for v in vs:
            if v not in node_to_triangles:
                node_to_triangles[v] = []
            node_to_triangles[v].append(k)

    areas_inner_nodes = []
    for surface in plane_surfaces:
        node_tags_inner, node_coords_inner, _ = gmsh.model.mesh.get_nodes(dim, surface, includeBoundary=False, returnParametricCoord=False)
        areas_inner_nodes.append(node_tags_inner)

    
    for iter in range(number_of_iterations):
        for surface, weight_function, inner_nodes in zip(plane_surfaces, weight_functions, areas_inner_nodes):
            for inner_node in inner_nodes:
                triangle_areas = gmsh.model.mesh.get_element_qualities(node_to_triangles[inner_node], qualityName='volume')

                all_coords = [[node_to_coords[node] for node in triangle_to_nodes[triangle]] for triangle in node_to_triangles[inner_node]]

                barycenters = np.sum(all_coords, axis=1) / 3
                weighted_areas = np.array([weight_function(barycenter) * triangle_area for barycenter, triangle_area in zip(barycenters, triangle_areas)])

                barycenter_area_product = barycenters * weighted_areas.reshape(-1, 1)
                new_coord = barycenter_area_product.sum(axis=0) / weighted_areas.sum()

                gmsh.model.mesh.set_node(inner_node, new_coord, [])
                node_to_coords[inner_node] = new_coord
        
        # move nodes on boundaries
        for curve in curve_loops:
            for inner_node in gmsh.model.mesh.get_nodes(1, curve, True)[0]:
                adjacent_triangles = node_to_triangles[inner_node]
                adjacent_triangles_areas = gmsh.model.mesh.get_element_qualities(adjacent_triangles, qualityName='volume')

                all_coords = [[node_to_coords[node] for node in triangle_to_nodes[triangle]] for triangle in adjacent_triangles]

                barycenters = np.sum(all_coords, axis=1) / 3
                barycenter_area_product = barycenters * adjacent_triangles_areas.reshape(-1, 1)
                new_coord = barycenter_area_product.sum(axis=0) / adjacent_triangles_areas.sum()

                new_coord = gmsh.model.get_closest_point(1, curve, new_coord)[0]

                gmsh.model.mesh.set_node(inner_node, new_coord, [])
                node_to_coords[inner_node] = new_coord
    
    gmsh.model.mesh.renumber_nodes()
    gmsh.model.mesh.renumber_elements()


    #gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    if fname is not None:
        gmsh.write(fname)

    gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    adaptation()
