from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import basix.ufl
import numpy as np


def solve(mesh_name, finite_element, k):
    mesh_type = mesh_name.split('.')[-1]
    if mesh_type == 'msh':
        domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    elif mesh_type == 'xdmf':
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")

    x = ufl.SpatialCoordinate(domain)
    u_analytic = lambda x: np.exp(x[0]*x[1])
    u_ufl = ufl.exp(x[0]*x[1])
    n = ufl.FacetNormal(domain)

    V = dolfinx.fem.functionspace(domain, finite_element)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = - ufl.div(k * ufl.grad(u_ufl))
    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx + ufl.dot(k*ufl.grad(u_ufl), n) * v * ufl.ds

    u_e = dolfinx.fem.Function(V)
    u_e.interpolate(u_analytic)
    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0)))
    bc = dolfinx.fem.dirichletbc(u_e, boundary_dofs)

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    V_compare = dolfinx.fem.functionspace(domain, basix.ufl.element("Lagrange", domain.topology.cell_name(), 4))
    u_e_compare = dolfinx.fem.Function(V_compare)
    u_e_compare.interpolate(u_analytic)

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
    # cell types: triangle quadrilateral
    import os
    mesh_name = os.path.join('meshes', 'msh', 'rectangle_1_quadrangle.msh')
    finite_element = basix.ufl.enriched_element([basix.ufl.element("Lagrange", 'quadrilateral', 1), basix.ufl.element("Bubble", 'quadrilateral', 3)])
    k = ufl.as_matrix([[1, 0],
                       [0, 1]])
    print(solve(mesh_name, finite_element, k))
