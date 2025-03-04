import sys
sys.path.append('mesh_generation')
import basic
import utility
import os
import os.path
import meshio
import quadrangle
import small_quadrangle
import quality
sys.path.append('mesh_generation/optimization')
import optimize
sys.path.append('mesh_generation/refinement')
import centroid
import circumcenter
import incenter
import orthocenter
import split_quadrangles
import uniform_split



meshes_dir = 'meshes'
data_dir = 'data'
meshes_msh_dir = os.path.join(meshes_dir, 'msh')
meshes_xdmf_dir = os.path.join(meshes_dir, 'xdmf')

for dir in (meshes_dir, data_dir, meshes_msh_dir, meshes_xdmf_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def convert(mesh_name, fname):
    mesh = meshio.read(mesh_name)
    meshio.write_points_cells(fname, mesh.points[:, :2], mesh.cells)
    print(f'{mesh_name:50}\tcells {mesh.cells[0].data.shape[0]}\tnodes {mesh.points.shape[0]}')


def get_names(name):
    triangle = f'{name}_triangle'
    optimized = f'{name}_triangle_optimized'
    quadrangle = f'{name}_quadrangle'
    small_quadrangle = f'{name}_small_quadrangle'
    small_triangle = f'{name}_small_triangle'

    msh, xdmf, data = [], [], []
    for i in (triangle, optimized, quadrangle, small_quadrangle, small_triangle):
        msh.append(os.path.join(meshes_msh_dir, f'{i}.msh'))
        xdmf.append(os.path.join(meshes_xdmf_dir, f'{i}.xdmf'))
        data.append(os.path.join(data_dir, f'{i}.npz'))
    
    return msh, xdmf, data


def run(name, nrefinements=0):
    polygon_points = (
        (0, 0), (0, 0.75), (1, 0.75), (1, 0),
    )
    polygon_mesh_size = 1

    msh, xdmf, data = get_names(f'{name}_{0}')

    while True:
        basic.polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, msh[0])
        start_max_angle, max_angle = optimize.optimize_max_angle(msh[0], msh[1], ui=True)
        print(start_max_angle, max_angle)
        if max_angle < 90:
            print(polygon_mesh_size, start_max_angle, max_angle)
            break
        polygon_mesh_size *= 0.9
    
    quadrangle.generate(msh[1], msh[2])
    small_quadrangle.generate(msh[1], msh[3])
    circumcenter.generate(msh[1], msh[4])

    for i, j, k in zip(msh, xdmf, data):
        convert(i, j)
        quality.save(i, k)

    previous_triangle = msh[1]
    
    for i in range(1, nrefinements + 1):
        msh, xdmf, data = get_names(f'{name}_{i}')
        uniform_split.uniform_split(previous_triangle, msh[0])
        start_max_angle, max_angle = optimize.change(msh[0], msh[0], ui=True)
        print(f'change after split {start_max_angle} -> {max_angle}')
        start_max_angle, max_angle = optimize.optimize_max_angle(msh[0], msh[1], min_count=1, ui=True)
        print(f'{start_max_angle} -> {max_angle}')
        quadrangle.generate(msh[1], msh[2])
        small_quadrangle.generate(msh[1], msh[3])
        circumcenter.generate(msh[1], msh[4])

        for i, j, k in zip(msh, xdmf, data):
            convert(i, j)
            quality.save(i, k)
                
        previous_triangle = msh[1]


if __name__ == '__main__':
    run('rectangle', 5)
