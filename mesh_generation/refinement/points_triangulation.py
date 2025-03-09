import gmsh
import numpy as np


# Триангуляция точек Делоне + Вороного (отличается ли от split_quadrangles? - нет)
def triangulate(quadrangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(quadrangle_mesh)

    # if ui:
    #     gmsh.fltk.run()

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    quadrangle_type = gmsh.model.mesh.get_element_type("Quadrangle", 1)

    one = np.uint64(1)

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            quad_tags, quads_nodes = gmsh.model.mesh.get_elements_by_type(quadrangle_type, surf_tag)
            
            node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(dim, surf_tag, includeBoundary=True, returnParametricCoord=False)
            node_coords = node_coords.reshape(-1, 3)[:, :2].flatten()
            triangulation = gmsh.model.mesh.triangulate(node_coords)
            # оно типо перенумерует ноды, но надо вернуться к глобальной
            triangulation = [node_tags[i - one] for i in triangulation]

            gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, [], triangulation)
            gmsh.model.mesh.remove_elements(dim, surf_tag, quad_tags)

    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()
    
    
    if ui:
        gmsh.fltk.run()
    
    gmsh.write(fname)

    gmsh.finalize()


if __name__ == '__main__':
    triangulate('meshes/msh/rectangle_1_quadrangle.msh', 'res.msh', True)
