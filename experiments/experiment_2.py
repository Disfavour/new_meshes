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


def run(name, nsets=1):
    polygon_points = (
        (0, 0), (0, 0.75), (1, 0.75), (1, 0),
    )
    polygon_mesh_size = 1

    msh, xdmf, data = get_names(f'{name}_{0}')

    #basic_triangle_mesh = 'meshes/test.msh'
    polygon_points = (
        (0, 0), (0, 0.75), (1, 0.75), (1, 0),
    )

    n = nsets
    polygon_mesh_size = 1

    mesh_names = [
        [
            f'{cur_name}_triangle',
            f'{cur_name}_quadrangle',
            f'{cur_name}_small_quadrangle',
            f'{cur_name}_circumcenter',
            f'{cur_name}_centroid',
            f'{cur_name}_incenter',
            f'{cur_name}_orthocenter',
            f'{cur_name}_split_quadrangles'
        ] for cur_name in (f'{name}_{i}' for i in range(1, n+1))
    ]
    mesh_names_msh = [[os.path.join(meshes_msh_dir, f'{mesh_name}.msh') for mesh_name in row] for row in mesh_names]
    mesh_names_xdmf = [[os.path.join(meshes_xdmf_dir, f'{mesh_name}.xdmf') for mesh_name in row] for row in mesh_names]
    mesh_names_npz = [[os.path.join(data_dir, f'{mesh_name}.npz') for mesh_name in row] for row in mesh_names]

    nodes, elements, mesh_sizes = [0], [0], []
    
    for i in range(n):
        while True:
            nodes_number , elements_number = basic.polygon_with_polygonal_holes(polygon_points, polygon_mesh_size, mesh_names_msh[i][0])
            # без оптимизации
            start_max_angle, max_angle = optimize.optimize_max_angle(mesh_names_msh[i][0], mesh_names_msh[i][0], max_count=0, ui=True)
            if max_angle < 85 and nodes_number != nodes[-1] and elements_number != elements[-1]:
                print(i, polygon_mesh_size, start_max_angle, max_angle)
                quadrangle.generate(mesh_names_msh[i][0], mesh_names_msh[i][1])
                small_quadrangle.generate(mesh_names_msh[i][0], mesh_names_msh[i][2])
                circumcenter.generate(mesh_names_msh[i][0], mesh_names_msh[i][3])
                centroid.generate(mesh_names_msh[i][0], mesh_names_msh[i][4])
                incenter.generate(mesh_names_msh[i][0], mesh_names_msh[i][5])
                orthocenter.generate(mesh_names_msh[i][0], mesh_names_msh[i][6])
                split_quadrangles.split(mesh_names_msh[i][1], mesh_names_msh[i][7])

                nodes.append(nodes_number)
                elements.append(elements_number)
                mesh_sizes.append(polygon_mesh_size)
                break
            polygon_mesh_size *= 0.99

        polygon_mesh_size /= 1.5 ** 0.5 # в 1.5 раз увеличится кол-во ячеек
    
    for row_msh, row_xdmf, row_npz in zip(mesh_names_msh, mesh_names_xdmf, mesh_names_npz):
        for msh, xdmf, npz in zip(row_msh, row_xdmf, row_npz):
            convert(msh, xdmf)
            quality.save(msh, npz)
    
    print(mesh_sizes)
    print(nodes)
    print(elements)


if __name__ == '__main__':
    run('rectangle', nsets=10)
