import gmsh
import utility
import numpy as np


qualities = [
    'minDetJac',    # the adaptively computed minimal Jacobian determinant
    'maxDetJac',    # the adaptively computed maximal Jacobian determinant
    'minSJ',        # sampled minimal scaled jacobien
    'minSICN',      # sampled minimal signed inverted condition number
    'minSIGE',      # sampled signed inverted gradient error
    'gamma',        # ratio of the inscribed to circumcribed sphere radius
    'innerRadius',
    'outerRadius',
    'minIsotropy',  # minimum isotropy measure
    'angleShape',   # angle shape measure
    'minEdge',      # minimum straight edge length
    'maxEdge',      # maximum straight edge length
    'volume',
]   # angles, min_angle, max_angle


def save(mesh, fname, ui=False):
    gmsh.initialize()

    if not ui:
        gmsh.option.setNumber("General.Terminal", 0)

    gmsh.open(mesh)

    element_types, element_tags, node_tags = gmsh.model.mesh.get_elements()

    data = {}
    for quality in qualities:
        q = gmsh.model.mesh.get_element_qualities(element_tags[0], quality)
        data[quality] = q
    
    angles = utility.get_all_angles()

    data['angles'] = angles

    data['min_angle'] = angles.min(axis=1)

    angles[angles == 180] = 0
    data['max_angle'] = angles.max(axis=1)

    np.savez_compressed(fname, **data)

    gmsh.finalize()


if __name__ == '__main__':
    save('meshes/msh/rectangle_1_quadrangle.msh', 'meshes/test_refine.msh', True)   # meshes/mesh_1_quadrangle.msh meshes/test.msh
