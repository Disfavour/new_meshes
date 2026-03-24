from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import numpy as np
import time
import sys


#  mpirun -n 6 --bind-to core --map-by socket python3 poisson.py

n = 1000
#domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral)
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle, diagonal=dolfinx.mesh.DiagonalType.right)

x = ufl.SpatialCoordinate(domain)
u_ufl = ufl.exp(x[0]*x[1])
u_a = lambda x: np.exp(x[0]*x[1])

V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

uD = dolfinx.fem.Function(V)
uD.interpolate(u_a)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k = ufl.as_matrix([
    [1, 0],
    [0, 1]
])

f = - ufl.div(k * ufl.grad(u_ufl))

a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}
# petsc_options={"ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000, "pc_type": "none"}

start_time = time.time()

uh = problem.solve()

elapsed_time = time.time() - start_time

V2 = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
uex = dolfinx.fem.Function(V2)
uex.interpolate(u_a)

L2_error = dolfinx.fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = dolfinx.fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = np.max(np.abs(uD.x.array-uh.x.array))
error_max = domain.comm.allreduce(error_max, op=MPI.MAX)

eh = uh - uex
error_H01 = dolfinx.fem.form(ufl.dot(ufl.grad(eh), ufl.grad(eh)) * ufl.dx)
error_H01 = np.sqrt(domain.comm.allreduce(dolfinx.fem.assemble_scalar(error_H01), op=MPI.SUM))

if MPI.COMM_WORLD.Get_rank() == 0:
    np.save(sys.argv[1], np.array((elapsed_time, error_L2, error_max, error_H01)))
    print(f'Elapsed time:\t{elapsed_time:.2f}') 
    print(f"Error L2:\t{error_L2:.2e}")
    print(f"Error max:\t{error_max:.2e}")
    print(f"Error H01:\t{error_H01:.2e}")
