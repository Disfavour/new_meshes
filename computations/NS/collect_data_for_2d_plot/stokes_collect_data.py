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


def solve(n, p, Re, fname):
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
    solver.convergence_criterion = "residual"
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

    n, converged = solver.solve(u)
    assert (converged)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Number of interations: {n:d} Converged {converged}")
        print(f'Solver time\t{time() - start_time}')
    
    assert MPI.COMM_WORLD.Get_size() == 1

    v = dolfinx.fem.Function(V)
    v.sub(0).interpolate(dolfinx.fem.Expression(u.sub(0).dx(1), V.element.interpolation_points()))
    v.sub(1).interpolate(dolfinx.fem.Expression(-u.sub(0).dx(0), V.element.interpolation_points()))

    np.savez(fname,
        dofs_coords=V.tabulate_dof_coordinates()[:, :2],
        psi_w=u.x.array,
        uv=v.x.array
    )


if __name__ == '__main__':
    solve(800, 2, 0, 'data/NS/stokes_n800_p2.npz')
    #solve(512, 3, 0, 'data/NS/stokes_n512_p3.npz')

    #solve(100, 2, 0, 'test.xdmf')
