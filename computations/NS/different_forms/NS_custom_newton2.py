from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
from dolfinx.io import XDMFFile, gmshio
import numpy as np

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


def calculate_residual(psi, w, psin, wn, Re, degree_raise=3):
    # residual = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
    #     Re*(
    #         (psi - psin).dx(1) * (w - wn).dx(0)
    #         - (psi - psin).dx(0) * (w - wn).dx(1)
    #     )*dx
    # ))
    # residual = np.abs(domain.comm.allreduce(residual, op=MPI.SUM))

    # Create higher order function space
    degree = psi.function_space.ufl_element().degree
    family = psi.function_space.ufl_element().family_name
    mesh = psi.function_space.mesh
    W = dolfinx.fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    psi_W = dolfinx.fem.Function(W)
    psi_W.interpolate(psi)

    psin_W = dolfinx.fem.Function(W)
    psin_W.interpolate(psin)

    w_W = dolfinx.fem.Function(W)
    w_W.interpolate(w)

    wn_W = dolfinx.fem.Function(W)
    wn_W.interpolate(wn)

    dif_psi = dolfinx.fem.Function(W)
    dif_psi.x.array[:] = psi_W.x.array - psin_W.x.array

    dif_w = dolfinx.fem.Function(W)
    dif_w.x.array[:] = w_W.x.array - wn_W.x.array

    residual = dolfinx.fem.Function(W)
    residual.interpolate(dolfinx.fem.Expression(
        Re*(
            (dif_w * dif_psi.dx(1)).dx(0)
            - (dif_w * dif_psi.dx(0)).dx(1)
        ),
        W.element.interpolation_points()
    ))
    error = dolfinx.fem.form(ufl.inner(residual, residual) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def solve(n, p, Re, fname):
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle)
    Re = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(Re))

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

    # - s.dx(0)*w*psi.dx(1) + s.dx(1)*w*psi.dx(0)
    F = dot(grad(psi), grad(r))*dx - w*r*dx - r*ds(1) \
        + dot(grad(w), grad(s))*dx + Re*(
            -(s.dx(0)*wn*psi.dx(1) + s.dx(0)*w*psin.dx(1) - s.dx(0)*wn*psin.dx(1))
            + s.dx(1)*wn*psi.dx(0) + s.dx(1)*w*psin.dx(0) - s.dx(1)*wn*psin.dx(0)
        )*dx(domain)
        
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
    
    errs_psi, errs_w, residuals = [], [], []
    for i in range(20):
        u = problem.solve()
        u.x.scatter_forward()
        #u.interpolate(dolfinx.fem.Expression(un + sigma*(u - un), V.element.interpolation_points()))
        
        psi, w = u.split()

        err_psi = error_L2(psi, psin)
        err_w = error_L2(w, wn)

        residual = calculate_residual(psi, w, psin, wn, Re)

        if domain.comm.rank == 0:
            print(err_psi, err_w, residual)

        un.x.array[:] = u.x.array

        errs_psi.append(err_psi)
        errs_w.append(err_w)
        residuals.append(residual)

    if domain.comm.rank == 0:
        print(f'Time {time() - start_time}')
        np.savez(fname,
            errs_psi=np.array(errs_psi),
            errs_w=np.array(errs_w),
            residuals=np.array(residuals)
        )
    
    return None


if __name__ == '__main__':
    #solve(100, 2, 100, None)
    Res = [500, 525, 550, 575, 600, 625, 650]
    ms = [128, 512]
    p = 2
    for m in ms:
            for Re in Res:
                solve(m, p, Re, f'data/NS/convergence2_m{m}_p{p}_Re{Re}.npz')
    
