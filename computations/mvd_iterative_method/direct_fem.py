from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
import numpy as np


def solve(mesh_name, k):
    domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    # mesh_type = mesh_name.split('.')[-1]
    # if mesh_type == 'msh':
    #     domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    # elif mesh_type == 'xdmf':
    #     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
    #         domain = xdmf.read_mesh(name="Grid")
    
    V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

    x = ufl.SpatialCoordinate(domain)
    #u_ufl =  ufl.exp(x[0]*x[1])# x[0]**2 + x[0]*x[1] #ufl.exp(x[0]*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(k)
    
    f = 1
    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx(domain)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs, V)

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_solver": "mumps"})
    u = problem.solve()

    return V.tabulate_dof_coordinates()[:, :2], u.x.array


if __name__ == '__main__':
    k = ufl.as_matrix([[1, 0],
                       [0, 1]])
    res = solve('meshes/rectangle/rectangle_11_quadrangle.msh', k)
    print(res)