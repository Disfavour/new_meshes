import gmsh
import numpy as np
import utility


# центр описанной окружности, пересечение серединных перпендикуляров
def generate(triangle_mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(triangle_mesh)

    triangle_type = gmsh.model.mesh.get_element_type("Triangle", 1)

    max_element_tag = gmsh.model.mesh.get_max_element_tag()

    for dim, physical_tag in gmsh.model.get_physical_groups(dim=2):
        for surf_tag in gmsh.model.get_entities_for_physical_group(dim, physical_tag):
            triangle_tags, triangle_nodes = gmsh.model.mesh.get_elements_by_type(triangle_type, surf_tag)
            center_tags = utility.add_triangle_centers(dim, surf_tag, triangle_nodes)
            
            small_triangle_tags, small_triangle_nodes = [], []
            for nodes, center in zip(triangle_nodes.reshape((-1, 3)), center_tags):
                for n1, n2 in zip(nodes, np.roll(nodes, -1)):
                    max_element_tag += 1
                    small_triangle_tags.append(max_element_tag)
                    small_triangle_nodes.extend((n1, n2, center))

            gmsh.model.mesh.add_elements_by_type(surf_tag, triangle_type, small_triangle_tags, small_triangle_nodes)
            gmsh.model.mesh.remove_elements(dim, surf_tag, triangle_tags)

    gmsh.model.mesh.renumber_elements()
    gmsh.model.mesh.renumber_nodes()

    gmsh.write(fname)

    if ui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == '__main__':
    generate('meshes/triangle.msh', 'meshes/new_small_triangle.msh', ui=True)
