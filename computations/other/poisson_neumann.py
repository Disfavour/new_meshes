from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh

from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad

import numpy as np
import pyvista

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
V = functionspace(mesh, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx


def u_exact(x):
    return 1 + x[0]**2 + 2 * x[1]**2


def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))


dofs_D = locate_dofs_geometrical(V, boundary_D)
u_bc = Function(V)
u_bc.interpolate(u_exact)
bc = dirichletbc(u_bc, dofs_D)


x = SpatialCoordinate(mesh)
f = Constant(mesh, default_scalar_type(-6))

# g = -4 * x[1]
# L = f * v * dx - g * v * ds

# если есть условия дирихле, то там автоматом обнулятся интегралы
import ufl
u_ufl = 1 + x[0]**2 + 2 * x[1]**2
n = ufl.FacetNormal(mesh) # ufl.as_vector([0, 1])
L = f * v * dx + ufl.dot(ufl.grad(u_ufl), n) * v * ds


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = functionspace(mesh, ("Lagrange", 2))
uex = Function(V2)
uex.interpolate(u_exact)
error_L2 = assemble_scalar(form((uh - uex)**2 * dx))
error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2, op=MPI.SUM))

u_vertex_values = uh.x.array
uex_1 = Function(V)
uex_1.interpolate(uex)
u_ex_vertex_values = uex_1.x.array
error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
print(f"Error_L2 : {error_L2:.2e}")
print(f"Error_max : {error_max:.2e}")
print(uh.x.array)
#print(uh.compute_point_values())