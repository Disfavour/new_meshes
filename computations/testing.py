import streamlit as st
import pandas as pd
from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

r"""
# FEniCSx и краевая задача Дирихле для уравнения Пуассона
"""


df = pd.DataFrame(
    [
       [1, 0],
       [0, 1]
   ]
)
edited_df = st.data_editor(df)
#print(edited_df.to_numpy())

def solve_poisson():
    #domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh('meshes/triangle_1.msh', MPI.COMM_WORLD, gdim=2)
    #domain, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh('meshes/triangle_1.msh', MPI.COMM_WORLD, gdim=2)

    u_ufl = ufl.exp(ufl.sqrt(x[0])*x[1])
    u_a = lambda x: np.exp(np.sqrt(x[0])*x[1])

    import meshio
    msh = meshio.read('meshes/triangle_3.msh')
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif  cell.type == "tetra":
            tetra_cells = cell.data

    triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": triangle_cells})
    meshio.write("mesh.xdmf", triangle_mesh)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh.xdmf', "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    V = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

    uD = dolfinx.fem.Function(V)
    uD.interpolate(lambda x: np.exp(np.sqrt(x[0])*x[1]))

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(edited_df.to_numpy().tolist())

    x = ufl.SpatialCoordinate(domain)
    #u_ufl = ufl.exp(ufl.sqrt(x[0])*x[1])
    f = - ufl.div(k * ufl.grad(u_ufl))

    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    V2 = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
    uex = dolfinx.fem.Function(V2)
    uex.interpolate(u_a)

    L2_error = dolfinx.fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    error_max = np.max(np.abs(uD.x.array-uh.x.array))

    eh = uh - uex
    error_H10 = dolfinx.fem.form(ufl.dot(ufl.grad(eh), ufl.grad(eh)) * ufl.dx)
    E_H10 = np.sqrt(domain.comm.allreduce(dolfinx.fem.assemble_scalar(error_H10), op=MPI.SUM))

    import pyvista
    from stpyvista import stpyvista

    from dolfinx import plot
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True, scalars="u")
    u_plotter.view_xy()
    # if not pyvista.OFF_SCREEN:
    #     u_plotter.show()
    #st.pyplot(plt.gcf())
    

    # import pyvista as pv
    # plotter = pv.Plotter(window_size=[400, 400])

    # ## Create a mesh
    # mesh = pv.Sphere(radius=1.0, center=(0, 0, 0))

    # ## Associate a scalar field to the mesh
    # x, y, z = mesh.cell_centers().points.T
    # mesh["My scalar"] = z

    # ## Add mesh to the plotter
    # plotter.add_mesh(
    #     mesh,
    #     scalars="My scalar",
    #     cmap="prism",
    #     show_edges=True,
    #     edge_color="#001100",
    #     ambient=0.2,
    # )

    # ## Some final touches
    # plotter.background_color = "white"
    # plotter.view_isometric()

    # ## Pass a plotter to stpyvista
    # stpyvista(plotter)

    stpyvista(u_plotter)

    f"Error_L2 : {error_L2:.2e}"
    f"Error_max : {error_max:.2e}"
    f"H01-error: {E_H10:.2e}"


if __name__ == '__main__':
    solve_poisson()
    