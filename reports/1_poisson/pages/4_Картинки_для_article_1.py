import streamlit as st
import pandas as pd
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import dolfinx.fem.petsc
import pandas as pd
import basix.ufl


name = 'rectangle'

basic_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
meshes_dir = os.path.join(basic_dir, 'meshes')
data_dir = os.path.join(basic_dir, 'data')
meshes_msh_dir = os.path.join(meshes_dir, 'msh')
meshes_xdmf_dir = os.path.join(meshes_dir, 'xdmf')

number_of_sets = 0
for i in range(1, 100):
    cur_name = f'{name}_{i}'
    if not os.path.exists(os.path.join(meshes_msh_dir, f'{cur_name}_triangle.msh')):
        break
    number_of_sets += 1

mesh_names = [
    [
        f'{name}_{i}_triangle',
        f'{name}_{i}_quadrangle',
        f'{name}_{i}_small_quadrangle',
        f'{name}_{i}_circumcenter',
        f'{name}_{i}_centroid',
        f'{name}_{i}_incenter',
        f'{name}_{i}_orthocenter',
        f'{name}_{i}_split_quadrangles',
        f'{name}_{i}_circumcenter_6',
        f'{name}_{i}_uniform_split'
    ] for i in range(1, number_of_sets + 1)
]
number_of_different_meshes = len(mesh_names[0])

mesh_names_xdmf = [[os.path.join(meshes_xdmf_dir, f'{mesh_name}.xdmf') for mesh_name in row] for row in mesh_names]
#mesh_names_npz = [[os.path.join(meshes_xdmf_dir, f'{mesh_name}.npz') for mesh_name in row] for row in mesh_names]

with st.sidebar:
    '#### Матрица $k$'
    df = pd.DataFrame(
        [
            [1, 0],
            [0, 1]
        ]
    )
    edited_df = st.data_editor(df)

    '#### Решение $u_a$'
    u_a_text = st.text_input(r'Решение $u_a$', 'np.exp(x[0]*x[1])', label_visibility='collapsed')
    u_ufl_text = u_a_text.replace('np.', 'ufl.')

    # '#### Конечные элементы'
    # finite_element = st.text_area(r'Тип конечных элементов', 'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)', label_visibility='collapsed')
    # #finite_element = eval(finite_element)

    '#### Конечные элементы для сравнения'
    finite_element2 = st.text_area(r'Тип конечных элементов', 'basix.ufl.element("Lagrange", domain.topology.cell_name(), 4)', label_visibility='collapsed')

    '#### Номер рисунка'
    option = st.selectbox("Номер рисунка", ("4", "5", '6'), label_visibility='collapsed')

    #'#### Сетки на графиках'
    mesh_options = (
        'Базовая',
        '4-угольная',
        '4-угольная 2',
        'Перпендикуляры',
        'Медианы',
        'Биссектрисы',
        'Высоты',
        'quad split',
        '6 треугольников',
        'Uniform split'
    )

    if option == "4":
        selected_options = ('Базовая', 'Перпендикуляры', 'Базовая')
        finite_elements = (
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
            'basix.ufl.enriched_element([basix.ufl.element("Lagrange", domain.topology.cell_name(), 1), basix.ufl.element("Bubble", domain.topology.cell_name(), 3)])'
        )
    elif option == "5":
        selected_options = ('Базовая', '6 треугольников', 'Базовая')
        finite_elements = (
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 2)',
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
            'basix.ufl.enriched_element([basix.ufl.element("Lagrange", domain.topology.cell_name(), 2), basix.ufl.element("Bubble", domain.topology.cell_name(), 3)])'
        )
    elif option == "6":
        selected_options = ('Базовая', '4-угольная', 'quad split')
        finite_elements = (
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
            'basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)',
        )
    selection = selected_options
    #selection = st.multiselect("Сетки на графиках", mesh_options, selected_options, label_visibility='collapsed')
    selection_idxs = [mesh_options.index(i) for i in selection]
    legend = ('а', 'б', 'в')


def solve_poisson(mesh_name, finite_element):
    u_a = lambda x: eval(u_a_text)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_name, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    V = dolfinx.fem.functionspace(domain, eval(finite_element))

    uD = dolfinx.fem.Function(V)
    uD.interpolate(u_a)

    tdim = domain.topology.dim
    # fdim = tdim - 1
    # domain.topology.create_connectivity(fdim, tdim)
    # boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    # boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    # bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)


    def boundary_D(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))


    dofs_D = dolfinx.fem.locate_dofs_geometrical(V, boundary_D)
    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_a)
    bc = dolfinx.fem.dirichletbc(u_bc, dofs_D)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = ufl.as_matrix(edited_df.to_numpy().tolist())

    x = ufl.SpatialCoordinate(domain)
    #
    u_ufl = eval(u_ufl_text)
    f = - ufl.div(k * ufl.grad(u_ufl))

    a = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    #L = f * v * ufl.dx
    n = ufl.FacetNormal(domain) # ufl.as_vector([0, 1])
    L = f * v * ufl.dx + ufl.dot(k*ufl.grad(u_ufl), n) * v * ufl.ds
    #L = f * v * ufl.dx + ufl.dot(ufl.dot(k, ufl.grad(u_ufl)), n) * v * ufl.ds

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    V2 = dolfinx.fem.functionspace(domain, eval(finite_element2))
    uex = dolfinx.fem.Function(V2)
    uex.interpolate(u_a)

    L2_error = dolfinx.fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    error_max = np.max(np.abs(uD.x.array-uh.x.array))
    error_max = domain.comm.allreduce(error_max, op=MPI.MAX)

    eh = uh - uex
    error_H10 = dolfinx.fem.form(ufl.dot(ufl.grad(eh), ufl.grad(eh)) * ufl.dx)
    E_H10 = np.sqrt(domain.comm.allreduce(dolfinx.fem.assemble_scalar(error_H10), op=MPI.SUM))

    cell_num = domain.topology.index_map(tdim).size_local
    node_num = domain.geometry.x.shape[0]

    dofs_num = V.dofmap.index_map.size_global

    return dofs_num, node_num, cell_num, error_L2, error_max, E_H10


complete = 0.0
progress_text = 'Вычисления'
progress_bar = st.progress(complete, text=progress_text)

data = []
for row in mesh_names_xdmf:
    data.append([])

    for i, finite_element in zip(selection_idxs, finite_elements):
        data[-1].append(solve_poisson(row[i], finite_element))

        complete += 1
        progress_bar.progress(complete / (number_of_sets * number_of_different_meshes), text=progress_text)

progress_bar.empty()

data = np.array(data)

tabs = st.tabs(('Неизвестные', 'Узлы', 'Ячейки'))

for tab, x, xlabel in zip(tabs, range(3), ('Неизвестные', 'Узлы', 'Ячейки')):
    with tab:
        # st.columns(3)
        columns = st.columns(2)
        for column, error, i in zip(columns, ('$L^2$', r'$L^{\infty}$', '$H_0^1$'), range(3, 6)):
            with column:
                plt.figure(figsize=(6.4, 3.6), dpi=300, tight_layout=True)
                plt.plot(data[:, :, x], data[:, :, i], '-o')

                plt.xlabel(xlabel)
                plt.ylabel(error)
                plt.grid()

                plt.xscale('log')
                plt.yscale('log')
                plt.legend(legend)

                st.pyplot(plt.gcf())
        
        with st.columns((0.25, 0.5, 0.25))[1]:
            plt.figure(figsize=(6.4, 3.6), dpi=300, tight_layout=True)
            plt.plot(data[:, :, x], data[:, :, 5], '-o')

            plt.xlabel(xlabel)
            plt.ylabel('$H_0^1$')
            plt.grid()

            plt.xscale('log')
            plt.yscale('log')
            plt.legend(legend)

            st.pyplot(plt.gcf())
