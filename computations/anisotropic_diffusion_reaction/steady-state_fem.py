from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
import numpy as np
import scipy.sparse


def solve(mesh_name, k, c=1):
    mesh_type = mesh_name.split('.')[-1]
    if mesh_type == 'msh':
        #print(dolfinx.io.gmsh.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2))
        #domain, cell_markers, facet_markers = dolfinx.io.gmsh.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
        domain = dolfinx.io.gmsh.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)[0]
    elif mesh_type == 'xdmf':
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
    
    x = ufl.SpatialCoordinate(domain)
    u_ufl = ufl.exp(1 + x[0]*x[1]) + ufl.exp(x[0]+3)# x[0]**2 + x[0]*x[1] #ufl.exp(x[0]*x[1])
    r = 1

    V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(k)
    
    f = - ufl.div(k * ufl.grad(u_ufl)) + r*u_ufl
    n = ufl.FacetNormal(domain)
    ub = ufl.dot(k * ufl.grad(u_ufl), n) + c*u_ufl

    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx + r*u*v * ufl.dx + c*u*v*ufl.ds(domain)
    L = f * v * ufl.dx + ub*v*ufl.ds

    u_e = dolfinx.fem.Function(V)
    u_e.interpolate(dolfinx.fem.Expression(u_ufl, V.element.interpolation_points))

    # tdim = domain.topology.dim
    # fdim = tdim - 1
    # domain.topology.create_connectivity(fdim, tdim)
    # boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    # boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    # bc = dolfinx.fem.dirichletbc(u_e, boundary_dofs)

    # problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="poisson_l")
    # u = problem.solve()

    bilinear_form = dolfinx.fem.form(a)
    linear_form = dolfinx.fem.form(L)

    A = dolfinx.fem.petsc.assemble_matrix(bilinear_form, bcs=[])
    A.assemble()
    b = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(linear_form))
    dolfinx.fem.petsc.assemble_vector(b, linear_form)

    indptr, indices, data = A.getValuesCSR()
    A_scipy = scipy.sparse.csr_matrix((data, indices, indptr), shape=A.getSize())
    print(f"Размер матрицы: {A_scipy.shape}")
    print(f"Количество ненулевых: {A_scipy.nnz}")
    tmp = A_scipy - A_scipy.T
    print(tmp.min(), tmp.max())

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    u = dolfinx.fem.Function(V)
    solver.solve(b, u.x.petsc_vec)

    V_compare = dolfinx.fem.functionspace(domain, ('Lagrange', 3))
    u_e_compare = dolfinx.fem.Function(V_compare)
    u_e_compare.interpolate(dolfinx.fem.Expression(u_ufl, V_compare.element.interpolation_points))

    L2_error = dolfinx.fem.form(ufl.inner(u - u_e_compare, u - u_e_compare) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    error_max = np.max(np.abs(u_e.x.array-u.x.array))
    error_max = domain.comm.allreduce(error_max, op=MPI.MAX)

    eh = u - u_e_compare
    error_H10 = dolfinx.fem.form(ufl.dot(ufl.grad(eh), ufl.grad(eh)) * ufl.dx)
    E_H10 = np.sqrt(domain.comm.allreduce(dolfinx.fem.assemble_scalar(error_H10), op=MPI.SUM))

    dofs_num = V.dofmap.index_map.size_global
    node_num = domain.geometry.x.shape[0]
    cell_num = domain.topology.index_map(domain.topology.dim).size_local

    return dofs_num, node_num, cell_num, error_L2, error_max, E_H10


if __name__ == '__main__':
    k = ufl.as_matrix([[1, 0.4],
                       [0.4, 2]])
    res = solve('meshes/rectangle/rectangle_8_quadrangle.msh', k, c=1)
    print(res)
    