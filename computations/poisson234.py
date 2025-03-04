from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import numpy as np
import ufl
import basix.ufl


k = ufl.as_matrix((
    (1, 0),
    (0, 1)
))

#domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh('mesh_1_triangle.msh', MPI.COMM_WORLD, gdim=2)

#domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5, dolfinx.mesh.CellType.triangle, diagonal=dolfinx.mesh.DiagonalType.left)
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5, dolfinx.mesh.CellType.quadrilateral)

#element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim, ))
element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2)
element1 = basix.ufl.element("Bubble", domain.topology.cell_name(), 3)
#el = element + element1
#exit()
#el2 = basix.ufl.enriched_element([element, element1])
#basix.ufl.mixed_element()
# ufl.TestFunctions()
# q = dolfinx.Function(VP)
# u, y = ufl.split(q)

#V = dolfinx.fem.functionspace(domain, el2)

#V = dolfinx.fem.functionspace(domain, basix.ufl.element("Hermite", domain.topology.cell_name(), 1))

V = dolfinx.fem.functionspace(domain, ('Bubble', 3))

#V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

uD = dolfinx.fem.Function(V)
uD.interpolate(lambda x: np.exp(x[0]*x[1]))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(domain)
u_ufl = ufl.exp(x[0]*x[1])
f = - ufl.div(k * ufl.grad(u_ufl))

a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
uex = dolfinx.fem.Function(V2)
uex.interpolate(lambda x: np.exp(x[0]*x[1]))

L2_error = dolfinx.fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = dolfinx.fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = np.max(np.abs(uD.x.array-uh.x.array))

eh = uh - uex
error_H10 = dolfinx.fem.form(ufl.dot(ufl.grad(eh), ufl.grad(eh)) * ufl.dx)
E_H10 = np.sqrt(domain.comm.allreduce(dolfinx.fem.assemble_scalar(error_H10), op=MPI.SUM))

if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
    print(f"H01-error: {E_H10:.2e}")

import pyvista

from dolfinx import plot
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()