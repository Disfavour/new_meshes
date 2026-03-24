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


    # r = ufl.TrialFunction(W)
    # rt = ufl.TestFunction(W)
    # a = r * rt * dx
    # L = Re*(dif_psi.dx(1) * dif_w.dx(0) - dif_psi.dx(0) * dif_w.dx(1)) * rt * dx
    # problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options = {
    #     "ksp_type": "preonly",
    #     "pc_type": "lu",
    #     "pc_factor_mat_solver_type": "mumps",
    # })
    # residual_project = problem.solve()
    # error = dolfinx.fem.form(ufl.inner(residual_project, residual_project) * ufl.dx)
    # error_local = dolfinx.fem.assemble_scalar(error)
    # error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    # residual_project = np.sqrt(error_global)
    # print(residual_project)


    residual = dolfinx.fem.Function(W)
    residual.interpolate(dolfinx.fem.Expression(
        Re*(
            dif_psi.dx(1) * dif_w.dx(0)
            - dif_psi.dx(0) * dif_w.dx(1)
        ),
        W.element.interpolation_points()
    ))
    error = dolfinx.fem.form(ufl.inner(residual, residual) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def solve(n, p, fname):
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle)
    Re = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0))

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

    Res, errs = [], []

    un_base = dolfinx.fem.Function(V)
    #u_local = dolfinx.fem.Function(V)
    step = 2000
    Re_global = 500
    #Re_local = Re_global
    n_iter = 0

    if domain.comm.rank == 0:
        print(f'Re_global {Re_global}')

    Re.value = Re_global
    errs_local_iter = []
    for i in range(10):
        u = problem.solve()
        u.x.scatter_forward()
        psi, w = u.split()
        errs_local_iter.append(error_L2(w, wn))

        if domain.comm.rank == 0:
            print(errs_local_iter[-1])

        un.x.array[:] = u.x.array
    
    Res.append(Re.value + 0)
    errs.append(errs_local_iter)
    n_iter += 1

    un_base.x.array[:] = u.x.array
    
    while True:

        if domain.comm.rank == 0:
            print(f'Re_global {Re_global}')

        while True:
            Re.value = Re_global + step
            un.x.array[:] = un_base.x.array

            if domain.comm.rank == 0:
                print(f'Re_global {Re_global} Re.value {Re.value} n_iter {n_iter}')
    
            errs_local_iter = []
            for i in range(10):
                u = problem.solve()
                u.x.scatter_forward()
                psi, w = u.split()
                errs_local_iter.append(error_L2(w, wn))

                if domain.comm.rank == 0:
                    print(errs_local_iter[-1])

                un.x.array[:] = u.x.array
            
            Res.append(Re.value + 0)
            errs.append(errs_local_iter)
            n_iter += 1

            if n_iter == 50:
                break
            
            if errs_local_iter[-1] < 1e-8:
                break

            step //= 2

            if domain.comm.rank == 0:
                print(f'Set Step {step}')
            
        if n_iter == 50:
            break
        
        Re_global = Re.value + 0    # тут присвоение как ссылка, а такое не надо
        un_base.x.array[:] = u.x.array
        

    if domain.comm.rank == 0:
        print(f'Time {time() - start_time}')
        np.savez(fname,
            Res=np.array(Res),
            errs=np.array(errs)
        )
    
    return None


if __name__ == '__main__':
    # m = 128
    m = 512
    p = 2
    solve(m, p, f'data/NS/bisection_m{m}_p{p}.npz')
