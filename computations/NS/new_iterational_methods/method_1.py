from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
from dolfinx.io import XDMFFile, gmshio
import numpy as np
from ufl import grad, dot, dx, ds
from time import time
from pathlib import Path


cache_dir = f"{str(Path.cwd())}/.cache"


def error_L2(uh, u_ex, degree_raise=3):
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


def solve(n, p, Re, sigma, fname):
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

    # psi.dx(1)*w.dx(0) - psi.dx(0)*w.dx(1)
    F = dot(grad(psi), grad(r))*dx - w*r*dx - r*ds(1) \
        + dot(grad(w), grad(s))*dx + Re*(
            psi.dx(1)*wn.dx(0)
            - psi.dx(0)*wn.dx(1)
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
        print(f'sigma {sigma}')
        start_time = time()
    
    errs = []
    for i in range(3000):
        iter_time = time()

        u = problem.solve()
        u.x.scatter_forward()

        u.x.array[:] = sigma * u.x.array + (1 - sigma) * un.x.array
        
        psi, w = u.split()
        err = error_L2(w, wn)

        if domain.comm.rank == 0:
            print(f'{i} err {err} time {time() - iter_time}')

        un.x.array[:] = u.x.array

        errs.append(err)

    if domain.comm.rank == 0:
        print(f'Time {time() - start_time}')
        np.savez(fname,
            errs=np.array(errs)
        )
    
    return None


if __name__ == '__main__':
    m = 128
    p = 2
    sigmas = [0.02, 0.03, 0.04, 0.05]
    Res = [20000]
    for Re in Res:
        for sigma in sigmas:
            solve(m, p, Re, sigma, f'data/NS/iterative_method_1_m{m}_p{p}_Re{Re}_s{sigma}.npz')
