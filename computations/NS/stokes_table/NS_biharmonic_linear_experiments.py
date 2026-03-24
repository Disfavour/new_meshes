from mpi4py import MPI
import dolfinx
import ufl
from ufl import grad, dot, dx, ds
import dolfinx.fem.petsc
import numpy as np
from time import time
from pathlib import Path
import sys
from dolfinx.io import XDMFFile, gmshio
import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

cache_dir = f"{str(Path.cwd())}/.cache"

mesh_name = sys.argv[1]
p = int(sys.argv[2])

domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
tdim = domain.topology.dim
fdim = tdim - 1

# domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle)
# tdim = domain.topology.dim
# fdim = tdim - 1

# facets = dolfinx.mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[1], 1)).astype(np.int32)
# facet_tag = dolfinx.mesh.meshtags(domain, fdim, facets, np.full_like(facets, 1, dtype=np.int32))
# ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

V = dolfinx.fem.functionspace(domain, ('Lagrange', p, (2,)))
psi, w = ufl.TrialFunctions(V)
s, r = ufl.TestFunctions(V)

a = (dot(grad(psi), grad(r)) - w * r + dot(grad(w), grad(s))) * dx
L = r * ds(1)

domain.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs, V.sub(0))

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    jit_options = {
        "cffi_extra_compile_args": ["-Ofast", "-march=native"],
        "cache_dir": cache_dir,
        "cffi_libraries": ["m"]
})

if MPI.COMM_WORLD.Get_rank() == 0:
    start_time = time()

u = problem.solve()

psi_vec, w_vec = map(lambda x: x.collapse().x.array, u.split())

psi_min = MPI.COMM_WORLD.allreduce(psi_vec.min(), op=MPI.MIN)
psi_max = MPI.COMM_WORLD.allreduce(psi_vec.max(), op=MPI.MAX)
w_min = MPI.COMM_WORLD.allreduce(w_vec.min(), op=MPI.MIN)
w_max = MPI.COMM_WORLD.allreduce(w_vec.max(), op=MPI.MAX)

if MPI.COMM_WORLD.Get_rank() == 0:
    # n, p, psi min, psi max
    print(mesh_name, p, psi_min, psi_max, w_min, w_max, time() - start_time, MPI.COMM_WORLD.Get_size())
