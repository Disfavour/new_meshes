from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
import numpy as np


def solve(mesh_name, k_np, u_np_text, c_np_text):
    mesh_type = mesh_name.split('.')[-1]
    if mesh_type == 'msh':
        domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    elif mesh_type == 'xdmf':
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")

    x = ufl.SpatialCoordinate(domain)
    #u_analytic = lambda x: np.exp(x[0]*x[1])
    u_analytic = lambda x: eval(u_np_text)
    #u_ufl = ufl.exp(x[0]*x[1])
    u_ufl = u_np_text.replace('np.', 'ufl.')
    #c_ufl = 0
    c_ufl = c_np_text.replace('np.', 'ufl.')

    V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(k_np)
    
    f = - ufl.div(k * ufl.grad(u_ufl)) + c_ufl*u_ufl
    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx + c_ufl*u_ufl * ufl.dx(domain)
    L = f * v * ufl.dx

    u_e = dolfinx.fem.Function(V)
    u_e.interpolate(u_analytic)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(u_e, boundary_dofs)

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    error_max = np.max(np.abs(u_e.x.array-u.x.array))
    error_max = domain.comm.allreduce(error_max, op=MPI.MAX)

    dofs_num = V.dofmap.index_map.size_global
    node_num = domain.geometry.x.shape[0]
    #print(domain.geometry.x)
    #print(V.dofmap.index_map)
    #print(V.tabulate_dof_coordinates())
    cell_num = domain.topology.index_map(domain.topology.dim).size_local

    #print(np.allclose(domain.geometry.x, V.tabulate_dof_coordinates()))


    #print(np.sort(u_e.x.array))
    #print(error_max)

    return dofs_num, node_num, cell_num, error_max, V.tabulate_dof_coordinates(), u.x.array


if __name__ == '__main__':
    k = ufl.as_matrix([[1, 0],
                       [0, 1]])
    solve('meshes/rectangle/rectangle_0_quadrangle.msh', k)