from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, plot, default_scalar_type
from dolfinx.fem import FunctionSpace, Function, Constant, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from ufl import (TestFunctions, TrialFunctions, dx, ds, inner, grad, Measure)
from petsc4py import PETSc
import dolfinx
import dolfinx.fem.petsc
import basix.ufl

# Параметры
f_value = 0.0
g_value = 1.0  # Значение нормальной производной на верхней границе

# Создание сетки
domain = mesh.create_unit_square(MPI.COMM_WORLD, 500, 500, mesh.CellType.triangle)

# Пространство: P2-P2
# P2 = finiteelement("Lagrange", domain.ufl_cell(), 2)
# element = MixedElement([P2, P2])
# W = FunctionSpace(domain, element)

# Solving linear variational problem.
# u min (по вершинам): -9.996045e-02
# u max (по вершинам): 2.195500e-06

#W = dolfinx.fem.functionspace(domain, ('Lagrange', 1, (2,)))

el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_mixed = basix.ufl.mixed_element([el_u, el_p])
W = dolfinx.fem.functionspace(domain, el_mixed)

(u, sigma) = TrialFunctions(W)
(v, tau) = TestFunctions(W)

f = Constant(domain, default_scalar_type(f_value))

# === 1. Условие Дирихле для u ===
def boundary(x):
    return np.full(x.shape[1], True, dtype=bool)  # Все граничные точки

boundary_dofs = locate_dofs_topological(W.sub(0), domain.topology.dim-1, 
                                       locate_entities_boundary(domain, domain.topology.dim-1, boundary))
bc_u = dirichletbc(default_scalar_type(0.0), boundary_dofs, W.sub(0))

# === 2. Верхняя граница (y = 1) ===
def top_boundary(x):
    return np.isclose(x[1], 1.0)

# Помечаем границы
fdim = domain.topology.dim - 1
boundary_facets = locate_entities_boundary(domain, fdim, top_boundary)
mt = mesh.meshtags(domain, fdim, boundary_facets, np.full(len(boundary_facets), 1, dtype=np.int32))

ds = Measure("ds", domain=domain, subdomain_data=mt)

# === 3. Неоднородное условие на ∂u/∂n ===
g = Constant(domain, default_scalar_type(g_value))

# Вариационная форма
a = (
    inner(grad(u), grad(tau)) * dx
    - sigma * tau * dx
    + inner(grad(sigma), grad(v)) * dx
)

# Правая часть: включает нагрузку и граничный вклад от ∂u/∂n = g
L = f * v * dx + g * tau * ds(1)

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc_u], petsc_options = {
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_monitor": None,
})
u = problem.solve()

print(u.x.array.real)
print(u.x.array.real[::2].min(), u.x.array.real[::2].max())
print(u.x.array.real[1::2].min(), u.x.array.real[1::2].max())

u_sol, sigma_sol = u.split()

print(u.x.array.real.shape, u_sol.x.array.real.shape, sigma_sol.x.array.real.shape)
print(u.x.array.real)
print(u_sol.x.array.real)

# --- Вывод глобальных минимум/максимум ---
u_vals = u_sol.x.array.real
u_min = np.min(u_vals)
u_max = np.max(u_vals)
print(f"u min: {u_min:.6e}")
print(f"u max: {u_max:.6e}")


V2 = dolfinx.fem.functionspace(domain, ('Lagrange', 1))

import pyvista
from dolfinx import plot
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V2)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

# u_grid.point_data["u"] = u.x.array.real[0::2]
# Интерполяция решений на grid
grid.point_data["u"] = u.x.array.real[::2] #u_sol.x.array
grid.point_data["sigma"] = u.x.array.real[1::2] #sigma_sol.x.array



# u_grid.set_active_scalars("u")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True, scalars="u")
# u_plotter.view_xy()
# u_plotter.show()

# Визуализация
plotter = pyvista.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_text("u(x, y)", font_size=12)
plotter.add_mesh(grid, scalars="u", show_edges=False)
plotter.add_scalar_bar("u", vertical=True)
plotter.view_xy()

plotter.subplot(0, 1)
plotter.add_text("σ(x, y) = -Δu", font_size=12)
plotter.add_mesh(grid, scalars="sigma", show_edges=False)
plotter.add_scalar_bar("σ", vertical=True)
plotter.view_xy()

plotter.show()
