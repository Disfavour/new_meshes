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


def solve(mesh_fname, Re, sigma, fname, p=2, max_iter=500):
    domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_fname, MPI.COMM_WORLD, gdim=2)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)

    Re = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(Re))

    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = dolfinx.fem.functionspace(domain, ('Lagrange', p, (2,)))
    un = dolfinx.fem.Function(V)
    psin, wn = un.split()
    psi, w = ufl.TrialFunctions(V)
    s, r = ufl.TestFunctions(V)

    # psi.dx(1)*w.dx(0) - psi.dx(0)*w.dx(1)
    F = dot(grad(psi), grad(r))*dx - w*r*dx - r*ds(1) \
        + dot(grad(w), grad(s))*dx + Re*(
            psi.dx(1)*wn.dx(0) + psin.dx(1)*w.dx(0) - psin.dx(1)*wn.dx(0)
            - (psi.dx(0)*wn.dx(1) + psin.dx(0)*w.dx(1) - psin.dx(0)*wn.dx(1))
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
    for i in range(max_iter):
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
    
    # save
    v = dolfinx.fem.Function(V)
    v.sub(0).interpolate(dolfinx.fem.Expression(u.sub(0).dx(1), V.element.interpolation_points()))
    v.sub(1).interpolate(dolfinx.fem.Expression(-u.sub(0).dx(0), V.element.interpolation_points()))

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
        np.savez(f'{fname}.npz',
            errs=np.array(errs),
            extreme_values=np.array((psi_min, psi_max, w_min, w_max, u_min, u_max, v_min, v_max))
        )

    V1 = dolfinx.fem.functionspace(domain, ('Lagrange', 1, (4,)))
    f = dolfinx.fem.Function(V1)
    f.sub(0).interpolate(u.sub(0))
    f.sub(1).interpolate(u.sub(1))
    f.sub(2).interpolate(v.sub(0))
    f.sub(3).interpolate(v.sub(1))

    with XDMFFile(domain.comm, f'{fname}.xdmf', "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(f)
    
    return None


if __name__ == '__main__':
    Re = 5500
    sigma = 0.09
    mesh_fname = 'meshes/NS/ellipse_256.msh'

    solve(mesh_fname, Re, sigma, None)
