from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
import numpy as np
from petsc4py import PETSc
import scipy


def solve(mesh_name, k):
    mesh_type = mesh_name.split('.')[-1]
    if mesh_type == 'msh':
        domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    elif mesh_type == 'xdmf':
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
    

    x = ufl.SpatialCoordinate(domain)
    u_ufl =  ufl.exp(x[0]*x[1]) # x[0]**2 + x[0]*x[1]
    c_ufl = 0

    V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(k)
    
    f = - ufl.div(k * ufl.grad(u_ufl))
    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx + c_ufl*u_ufl * ufl.dx(domain)
    L = f * v * ufl.dx

    u_e = dolfinx.fem.Function(V)
    u_e.interpolate(dolfinx.fem.Expression(u_ufl, V.element.interpolation_points()))

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(u_e, boundary_dofs)

    a = dolfinx.fem.form(a)
    b = dolfinx.fem.form(L)
    L = dolfinx.fem.form(L)

    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    A1 = A.copy()
    #b = dolfinx.fem.petsc.create_vector(dolfinx.fem.form(L))
    b = dolfinx.fem.petsc.assemble_vector(L)

    rows, cols, values = A1.getValuesCSR()
    A_scipy = scipy.sparse.csr_matrix((values, cols, rows))

    dof_coords = V.tabulate_dof_coordinates()

    b1 = b.array.copy()

    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.petsc.apply_lifting(b, [a], [[bc]])
    b2 = b.array.copy()
    #b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])
    b3 = b.array.copy()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    u = dolfinx.fem.Function(V)
    # Solve linear problem
    solver.solve(b, u.x.petsc_vec)

    
    #problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    #u = problem.solve()

    V_compare = dolfinx.fem.functionspace(domain, ('Lagrange', 3))
    u_e_compare = dolfinx.fem.Function(V_compare)
    u_e_compare.interpolate(dolfinx.fem.Expression(u_ufl, V_compare.element.interpolation_points()))

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
    #print(error_L2)

    return u.x.array, dof_coords[:, :2], A_scipy.toarray(), b1, b2, b3


if __name__ == '__main__':
    k = ufl.as_matrix([[1, 0],
                       [0, 1]])
    res = solve('meshes/rectangle/old_rectangle_0_triangle.msh', k)
    print(res)