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


def error_L2(uh, u_ex, degree_raise=3):
    # err_psi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dot(psi - psin, psi - psin)*dx))
    # err_psi = np.sqrt(domain.comm.allreduce(err_psi, op=MPI.SUM))

    # err_w = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dot(w - wn, w - wn)*dx))
    # err_w = np.sqrt(domain.comm.allreduce(err_w, op=MPI.SUM))

    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = dolfinx.fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = dolfinx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dolfinx.fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dolfinx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def solve(n, p, Re_max):
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle)
    Re = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0))

    best_steps = [500, 1000, 2500, 7000, 12000, 15500, 19000, 20000]
    assert Re_max <= best_steps[-1]
    assert MPI.COMM_WORLD.Get_size() == 1

    tdim = domain.topology.dim
    fdim = tdim - 1

    facets = dolfinx.mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[1], 1)).astype(np.int32)
    facet_tag = dolfinx.mesh.meshtags(domain, fdim, facets, np.full_like(facets, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    V = dolfinx.fem.functionspace(domain, ('Lagrange', p, (2,)))
    un = dolfinx.fem.Function(V)
    psin, wn = un.split()
    psi, w = ufl.TrialFunctions(V)
    s, r = ufl.TestFunctions(V)

    F = dot(grad(psi), grad(r))*dx - w*r*dx - r*ds(1) \
        + dot(grad(w), grad(s))*dx + Re*(
            psi.dx(1)*wn.dx(0) + psin.dx(1)*w.dx(0) - psin.dx(1)*wn.dx(0)   # psi.dx(1)*w.dx(0)
            - (psi.dx(0)*wn.dx(1) + psin.dx(0)*w.dx(1) - psin.dx(0)*wn.dx(1))    # - psi.dx(0)*w.dx(1)
        )*s*dx(domain)
        
    a, L = ufl.system(F)

    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs, V.sub(0))

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }, jit_options = {
        "cffi_extra_compile_args": ["-Ofast", "-march=native"],
        "cache_dir": cache_dir,
        "cffi_libraries": ["m"]
    })

    if domain.comm.rank == 0:
        start_time = time()

    for Re_current in best_steps:
        Re.value = min(Re_max, Re_current)
        #un.x.array[:] = un_base.x.array
        print(Re.value)

        Re_time = time()
        while True:
            iter_time = time()
            u = problem.solve()
            #u.x.scatter_forward()
            psi, w = u.split()

            #err = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(dot(w - wn, w - wn)*dx)))
            err = error_L2(w, wn)

            print(err, time() - iter_time)

            un.x.array[:] = u.x.array

            if err < 1e-8:
                break
        
        print(f'Re_time {time() - Re_time}')

        if np.isclose(Re.value, 1000) or np.isclose(Re.value, Re_max):
            v = dolfinx.fem.Function(V)
            v.sub(0).interpolate(dolfinx.fem.Expression(u.sub(0).dx(1), V.element.interpolation_points()))
            v.sub(1).interpolate(dolfinx.fem.Expression(-u.sub(0).dx(0), V.element.interpolation_points()))

            np.savez(f'data/NS/Re{int(Re.value)}_n{m}_p{p}.npz',
                dofs_coords=V.tabulate_dof_coordinates()[:, :2],
                psi_w=u.x.array,
                uv=v.x.array
            )

        if np.isclose(Re.value, Re_max):
            break
    
    print(f'Time {time() - start_time}')
    
    return None


if __name__ == '__main__':
    m = 800
    p = 2
    Re_max = 10000
    solve(m, p, Re_max)
