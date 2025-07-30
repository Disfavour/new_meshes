import streamlit as st
import sys
sys.path.append('computations/elliptic')
import mvd
import fem
import ufl
import pandas as pd
import sympy
import gmsh

gmsh.initialize(["-nopopup", "-"] )
mesh_fnames = [f'meshes/rectangle/rectangle_{i}_quadrangle.xdmf' for i in range(10)]
i = st.slider('mesh', 0, 9)

x, y = sympy.symbols('x y')

with st.sidebar:
    '#### Матрица $k$'
    df = pd.DataFrame(
        [
            [1, 0],
            [0, 1]
        ]
    )
    edited_df = st.data_editor(df)
    k_np = edited_df.to_numpy()

    '#### Решение $u$'
    u_np_text = st.text_input(r'Решение $u_e$', 'np.exp(x[0]*x[1])', label_visibility='collapsed')
    '#### $c$'
    c_np_text = st.text_input(r'$c$', '0', label_visibility='collapsed')

    u_text = u_np_text.replace('x[0]', 'x').replace('x[1]', 'y')
    c_text = c_np_text.replace('x[0]', 'x').replace('x[1]', 'y')

    u = eval(u_text.replace('np.', 'sympy.'))
    c = eval(c_text.replace('np.', 'sympy.'))
    k = sympy.Matrix(k_np)

    grad_u = sympy.Matrix([u.diff(x), u.diff(y)])
    flux = sympy.Matrix(k) * grad_u
    div_flux = flux[0].diff(x) + flux[1].diff(y)
    f = -div_flux + c * u
    
    u_np = sympy.lambdify([x, y], u, "numpy")
    f_np = sympy.lambdify([x, y], f, "numpy")
    c_np = sympy.lambdify([x, y], c, "numpy")


k = ufl.as_matrix([[1, 0],
                    [0, 1]])

dofs_num, node_num, cell_num, error_max, node_coords_fem, u_fem = fem.solve('meshes/rectangle/rectangle_0_quadrangle.xdmf', k_np, u_np_text, c_np_text)

#st.pyplot()