import gmsh
import numpy as np


dim = 2
z = 0


def polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, fname, holes_points=[], holes_mesh_sizes=[], ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    polygon_points_ = []
    for x, y in polygon_points:
        polygon_points_.append(gmsh.model.geo.add_point(x, y, z, polygon_mesh_size))
    
    holes_points_ = []
    for hole_points, hole_mesh_size in zip(holes_points, holes_mesh_sizes):
        holes_points_.append([])
        for x, y in hole_points:
            holes_points_[-1].append(gmsh.model.geo.add_point(x, y, z, hole_mesh_size))

    polygon_lines = []
    for point1, point2 in zip(polygon_points_, np.roll(polygon_points_, -1)):
        polygon_lines.append(gmsh.model.geo.add_line(point1, point2))
    
    holes_lines = []
    for hole_points in holes_points_:
        holes_lines.append([])
        for point1, point2 in zip(hole_points, np.roll(hole_points, -1)):
            holes_lines[-1].append(gmsh.model.geo.add_line(point1, point2))
    
    polygon_curve_loop = gmsh.model.geo.add_curve_loop(polygon_lines)

    holes_curve_loops = []
    for hole_lines in holes_lines:
        holes_curve_loops.append(gmsh.model.geo.add_curve_loop(hole_lines))

    plane_surface = gmsh.model.geo.add_plane_surface([polygon_curve_loop, *holes_curve_loops])

    gmsh.model.geo.synchronize()

    physgroup = gmsh.model.addPhysicalGroup(dim, [plane_surface])

    gmsh.model.mesh.generate(dim)

    gmsh.write(fname)

    if ui:
        gmsh.fltk.run()

    node_tags, coord, parametric_coord = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))

    gmsh.finalize()

    return node_tags.size, element_tags.size


# Первое ребро общее (mesh_size_shared для общего ребра)
def rectangle_with_subdomains(subdomain1_points, subdomain2_points, mesh_size, mesh_size_shared, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    subdomain1_points_, subdomain2_points_ = [], []
    for subdomain_points, subdomain_points_ in zip((subdomain1_points, subdomain2_points), (subdomain1_points_, subdomain2_points_)):
        x, y = subdomain_points[0]
        subdomain_points_.append(gmsh.model.geo.add_point(x, y, z, mesh_size_shared))
        for x, y in subdomain_points[1:]:
            subdomain_points_.append(gmsh.model.geo.add_point(x, y, z, mesh_size))
        
    subdomain1_lines, subdomain2_lines = [], []
    for subdomain_points, subdomain_lines in zip((subdomain1_points_, subdomain2_points_), (subdomain1_lines, subdomain2_lines)):
        for point1, point2 in zip(subdomain_points, np.roll(subdomain_points, -1)):
            subdomain_lines.append(gmsh.model.geo.add_line(point1, point2))

    subdomain1_curve_loop = gmsh.model.geo.add_curve_loop(subdomain1_lines)
    subdomain2_curve_loop = gmsh.model.geo.add_curve_loop(subdomain2_lines)

    subdomain1_plane_surface = gmsh.model.geo.add_plane_surface([subdomain1_curve_loop])
    subdomain2_plane_surface = gmsh.model.geo.add_plane_surface([subdomain2_curve_loop])

    gmsh.model.geo.synchronize()

    subdomain1_physgroup = gmsh.model.addPhysicalGroup(dim, [subdomain1_plane_surface])
    subdomain2_physgroup = gmsh.model.addPhysicalGroup(dim, [subdomain2_plane_surface])

    #subdomain1_physgroup = gmsh.model.addPhysicalGroup(dim, [subdomain1_plane_surface, subdomain2_plane_surface])

    gmsh.model.mesh.generate(dim)

    gmsh.write(fname)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    #generate(0, 0, 1, 0.75, 0.2, 'meshes/triangle.msh', ui=True)
    #generate_with_subdomains(0, 0, 1, 0.75, 0.4, 0.6, 0.2, 0.2, 'meshes/triangle_subdomains.msh', ui=True)
    #generate_with_subdomains_v2(0, 0, 1, 0.75, 0.4, 0.6, 0.2, 0.2, 'meshes/triangle_subdomains.msh', ui=True)
    #polygon_with_polygonal_holes(((0, 0), (0, 1), (2, 1), (2, 0)), (((0.2, 0.2), (0.5, 0.8), (0.8, 0.2)), ((1.2, 0.2), (1.2, 0.8), (1.8, 0.8), (1.8, 0.2))), 0.2, (0.05, 0.05), 'meshes/new.msh', ui=True)
    #polygon_with_polygonal_holes(((0, 0), (0, 1), (2, 1), (2, 0)), 0.2, 'meshes/new.msh', ui=True)

    import math
    basic_triangle_mesh = 'meshes/test.msh'
    # polygon_points = (
    #     (0, 0), (0, 0.75), (1, 0.75), (1, 0),
    # )
    polygon_points = (
        (1, 0), (1, 0.75), (0, 0.75), (0, 0)
    )
    polygon_mesh_size = 1#0.2
    mesh_sizes = []
    import optimization
    for i in range(15):
        while True:
            polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, basic_triangle_mesh)
            start_max_angle, max_angle = optimization.optimize_max_angle(basic_triangle_mesh, 'meshes/test_optimized.msh', ui=True)
            if max_angle < 80:
                i += 1
                print(i, polygon_mesh_size, start_max_angle, max_angle)
                break
            polygon_mesh_size *= 0.99
        
        polygon_mesh_size /= 1.5 ** 0.5
