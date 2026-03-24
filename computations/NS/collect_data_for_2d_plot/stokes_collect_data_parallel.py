from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
from dolfinx.io import XDMFFile, gmshio
import numpy as np
from petsc4py import PETSc
import dolfinx.nls.petsc

from ufl import grad, dot, dx, ds

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

import scipy
from time import time
from pathlib import Path
import sympy
import matplotlib.ticker as ticker
import gmsh


cache_dir = f"{str(Path.cwd())}/.cache"


def solve(n, p, Re):
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle)
    Re = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(Re))

    tdim = domain.topology.dim
    fdim = tdim - 1

    facets = dolfinx.mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[1], 1)).astype(np.int32)
    facet_tag = dolfinx.mesh.meshtags(domain, fdim, facets, np.full_like(facets, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    V = dolfinx.fem.functionspace(domain, ('Lagrange', p, (2,)))
    u = dolfinx.fem.Function(V)
    psi, w = ufl.split(u)
    s, r = ufl.TestFunctions(V)

    F = dot(grad(psi), grad(r))*dx - w*r*dx - r*ds(1) \
        + dot(grad(w), grad(s))*dx + Re*(psi.dx(1)*w.dx(0) - psi.dx(0)*w.dx(1))*s*dx(domain)

    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs, V.sub(0))

    if MPI.COMM_WORLD.Get_rank() == 0:
        start_time = time()
    
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs=[bc], jit_options={
        "cffi_extra_compile_args": ["-Ofast", "-march=native"],
        "cache_dir": cache_dir,
        "cffi_libraries": ["m"]
    })

    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "residual" # incremental residual
    solver.rtol = 1e-9
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'Assemble time\t{time() - start_time}')
        start_time = time()

    iters, converged = solver.solve(u)
    assert (converged)
    u.x.scatter_forward()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Number of interations: {iters:d} Converged {converged}")
        print(f'Solver time\t{time() - start_time}')
    
    v = dolfinx.fem.Function(V)
    v.sub(0).interpolate(dolfinx.fem.Expression(u.sub(0).dx(1), V.element.interpolation_points()))
    v.sub(1).interpolate(dolfinx.fem.Expression(-u.sub(0).dx(0), V.element.interpolation_points()))

    V1 = dolfinx.fem.functionspace(domain, ('Lagrange', 1, (4,)))
    f = dolfinx.fem.Function(V1)
    f.sub(0).interpolate(u.sub(0))
    f.sub(1).interpolate(u.sub(1))
    f.sub(2).interpolate(v.sub(0))
    f.sub(3).interpolate(v.sub(1))

    with XDMFFile(domain.comm, f'data/NS/Re{int(Re.value)}_n{n}_p{p}.xdmf', "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(f)

    psi_vec, w_vec = map(lambda x: x.collapse().x.array, u.split())
    psi_min = domain.comm.allreduce(psi_vec.min(), op=MPI.MIN)
    psi_max = domain.comm.allreduce(psi_vec.max(), op=MPI.MAX)
    w_min = domain.comm.allreduce(w_vec.min(), op=MPI.MIN)
    w_max = domain.comm.allreduce(w_vec.max(), op=MPI.MAX)

    u_vec, v_vec = map(lambda x: x.collapse().x.array, v.split())
    u_min = domain.comm.allreduce(u_vec.min(), op=MPI.MIN)
    u_max = domain.comm.allreduce(u_vec.max(), op=MPI.MAX)
    v_min = domain.comm.allreduce(v_vec.min(), op=MPI.MIN)
    v_max = domain.comm.allreduce(v_vec.max(), op=MPI.MAX)
    
    if domain.comm.rank == 0:
        np.savez(f'data/NS/extreme_values_Re{int(Re.value)}_n{n}_p{p}.npz',
            extreme_values=np.array((psi_min, psi_max, w_min, w_max, u_min, u_max, v_min, v_max))
        )
    
    return None


if __name__ == '__main__':
    n = 800
    p = 2
    Re = 0
    solve(n, p, Re)
