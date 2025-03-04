import gmsh
import utility
import os
import Laplace
import Lloyd
import numpy as np
import time


def optimize_max_angle(triangle_mesh, fname, min_count=0, max_count=10, atol=0.1, niter=1, ui=False):
    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)

    best_mesh = 'best.msh'

    gmsh.open(triangle_mesh)
    gmsh.write(fname)

    optimization_sequence = []
    max_angles = [utility.get_max_angle()]

    gmsh.clear()

    for i in range(max_count):
        best_angle = 360
        best_method = None

        for method in ('Relocate2D', 'Laplace2D', 'UntangleMeshGeometry'):
            gmsh.open(fname)
            start_time = time.time()
            gmsh.model.mesh.optimize(method=method, niter=niter)
            max_angle = utility.get_max_angle()

            if ui: print(f'{method} ({time.time() - start_time:.2f} sec): {max_angles[-1]:.2f} -> {max_angle:.2f}')

            if max_angle < best_angle:
                gmsh.write(best_mesh)
                best_angle = max_angle
                best_method = method

            gmsh.clear()
        

        for method, method_name in zip((Laplace.simultaneous_version, Laplace.sequential_version, Lloyd.simultaneous_version, Lloyd.sequential_version),
                                       ('Laplace simultaneous', 'Laplace sequential', 'Lloyd simultaneous', 'Lloyd sequential')):
            gmsh.open(fname)
            start_time = time.time()
            method(niter)
            max_angle = utility.get_max_angle()

            if ui: print(f'{method_name} ({time.time() - start_time:.2f} sec): {max_angles[-1]:.2f} -> {max_angle:.2f}')

            if max_angle < best_angle:
                gmsh.write(best_mesh)
                best_angle = max_angle
                best_method = method_name

            gmsh.clear()
        
        if ui: print(best_angle, best_method, '\n')
        
        if max_angles[-1] - best_angle > atol or i < min_count:
            max_angles.append(best_angle)
            optimization_sequence.append(best_method)

            gmsh.open(best_mesh)
            gmsh.write(fname)
            gmsh.clear()
        else:
            break
    
    if ui:
        print(optimization_sequence)
        print(max_angles)
        
    if os.path.exists(best_mesh):
        os.remove(best_mesh)

    gmsh.finalize()

    return max_angles[0], max_angles[-1]


def change(triangle_mesh, fname, min_count=0, max_count=10, atol=0.1, niter=1, ui=False):
    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)

    best_mesh = 'best.msh'

    gmsh.open(triangle_mesh)
    gmsh.write(fname)

    optimization_sequence = []
    max_angles = [utility.get_max_angle()]
    differences = [0]

    node_tags, node_coords_start, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    node_coords_start = node_coords_start.reshape(-1, 3)

    gmsh.clear()

    for i in range(max_count):
        best_angle = 360
        best_method = None
        best_difference = 0

        for method in ('Relocate2D', 'Laplace2D', 'UntangleMeshGeometry'):
            gmsh.open(fname)
            start_time = time.time()
            gmsh.model.mesh.optimize(method=method, niter=niter)
            max_angle = utility.get_max_angle()

            node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
            node_coords = node_coords.reshape(-1, 3)
            difference = np.linalg.norm(node_coords - node_coords_start, axis=1).sum()

            if ui: print(f'{method} ({time.time() - start_time:.2f} sec): {max_angles[-1]:.2f} -> {max_angle:.2f} difference {difference}')

            if difference > best_difference:
                gmsh.write(best_mesh)
                best_angle = max_angle
                best_method = method
                best_difference = difference

            gmsh.clear()
        

        for method, method_name in zip((Laplace.simultaneous_version, Laplace.sequential_version, Lloyd.simultaneous_version, Lloyd.sequential_version),
                                       ('Laplace simultaneous', 'Laplace sequential', 'Lloyd simultaneous', 'Lloyd sequential')):
            gmsh.open(fname)
            start_time = time.time()
            method(niter)
            max_angle = utility.get_max_angle()

            node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
            node_coords = node_coords.reshape(-1, 3)
            difference = np.linalg.norm(node_coords - node_coords_start, axis=1).sum()

            if ui: print(f'{method_name} ({time.time() - start_time:.2f} sec): {max_angles[-1]:.2f} -> {max_angle:.2f} difference {difference}')

            if difference > best_difference:
                gmsh.write(best_mesh)
                best_angle = max_angle
                best_method = method_name
                best_difference = difference

            gmsh.clear()
        
        if ui: print(best_difference, best_angle, best_method, '\n')
        
        if best_difference - differences[-1] > atol or i < min_count:
            max_angles.append(best_angle)
            optimization_sequence.append(best_method)
            differences.append(best_difference)

            gmsh.open(best_mesh)
            gmsh.write(fname)
            gmsh.clear()
        else:
            break
    
    if ui:
        print(optimization_sequence)
        print(max_angles)
        print(differences)
        
    if os.path.exists(best_mesh):
        os.remove(best_mesh)

    gmsh.finalize()

    return max_angles[0], max_angles[-1]


if __name__ == '__main__':
    change('meshes/msh/rectangle_4_triangle.msh', 'triangle_optimized.msh', min_count=0, max_count=100, atol=0.1, ui=True)
