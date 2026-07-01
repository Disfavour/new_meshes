import dolfinx
from mpi4py import MPI
import ufl
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import pyvista
import basix.ufl



def solve(mesh_fname, element, K, r, c):
    mesh_data = gmshio.read_from_msh(mesh_fname, MPI.COMM_WORLD, gdim=2)
    domain = mesh_data.mesh
    assert mesh_data.cell_tags is not None
    cell_markers = mesh_data.cell_tags
    assert mesh_data.facet_tags is not None
    facet_markers = mesh_data.facet_tags
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)

    K = ufl.as_matrix(K)

    V = dolfinx.fem.functionspace(domain, element)

    x, y = ufl.SpatialCoordinate(domain)
    ue = ufl.exp(ufl.sqrt(1 + x) * y)
    expr = dolfinx.fem.Expression(ue, V.element.interpolation_points)

    n = ufl.FacetNormal(domain)
    
    b = ufl.dot(ufl.dot(K, ufl.grad(ue)), n) + c*ue

    uD = dolfinx.fem.Function(V)
    uD.interpolate(expr)


    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    #bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = -ufl.div(ufl.dot(K, ufl.grad(ue))) + r*ue

    a = c*u*v * ds + ufl.dot(ufl.grad(v), ufl.dot(K, ufl.grad(u))) * ufl.dx + r*u*v * ufl.dx
    L = f * v * ufl.dx + b*v * ds

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="Poisson",
    )
    uh = problem.solve()

    #V2 = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
    V2 = dolfinx.fem.functionspace(domain, ("CR", 1))
    uex = dolfinx.fem.Function(V2, name="u_exact")
    expr = dolfinx.fem.Expression(ue, V2.element.interpolation_points)
    uex.interpolate(expr)

    L2_error = dolfinx.fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    error_max_local = np.max(np.abs(uD.x.array - uh.x.array))
    error_max = domain.comm.allreduce(error_max_local, op=MPI.MAX)

    num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    num_nodes_global = domain.geometry.index_map().size_global
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    #print(num_cells_global, num_nodes_global, V.dofmap.index_map.size_global, V.dofmap.index_map_bs)

    # domain.topology.create_connectivity(tdim, tdim)
    # topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim)
    # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # plotter.view_xy()
    # plotter.show()

    return num_dofs_global, error_L2, error_max


if __name__ == '__main__':
    K = np.array((
        (1, 0),
        (0, 1)
    ))
    mesh_fname = f'meshes/rectangle/rectangle_{5}_triangle.msh'
    res = solve(mesh_fname, ('CR', 1), K, 1, 1)
    print(res)
    exit()

    data = []
    for i in range(1, 11):
        for finite_element in (('Lagrange', 1), basix.ufl.element(basix.ElementFamily.CR, basix.CellType.quadrilateral, 1)):
            mesh_fname = f'meshes/rectangle/rectangle_{i}_quadrangle.msh'
            res = solve(mesh_fname, finite_element)
            print(res)
            
            data.append(res)
    
    data = np.array(data)
    print(data[:, 2:])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8), layout='constrained')

    axs[0, 0].plot(data[::2, 0], data[::2, 3], '-o')
    axs[0, 0].plot(data[1::2, 0], data[1::2, 3], '-o')
    axs[0, 0].set_ylabel(r'$L_2$')
    axs[0, 0].set_xlabel('dofs')

    axs[0, 1].plot(data[::2, 0], data[::2, 4], '-o')
    axs[0, 1].plot(data[1::2, 0], data[1::2, 4], '-o')
    axs[0, 1].set_ylabel(r'$L_\infty$')
    axs[0, 1].set_xlabel('dofs')

    axs[1, 0].plot(data[::2, 1], data[::2, 3], '-o')
    axs[1, 0].plot(data[1::2, 1], data[1::2, 3], '-o')
    axs[1, 0].set_ylabel(r'$L_2$')
    axs[1, 0].set_xlabel('nodes')

    axs[1, 1].plot(data[::2, 1], data[::2, 4], '-o')
    axs[1, 1].plot(data[1::2, 1], data[1::2, 4], '-o')
    axs[1, 1].set_ylabel(r'$L_\infty$')
    axs[1, 1].set_xlabel('nodes')

    for ax in axs.flat:
        ax.grid()
        ax.loglog()
        ax.legend(('Lagrage', 'CR'))
    
    plt.savefig('compare_P_CR_quadrangle.pdf', transparent=True)
    plt.show()

