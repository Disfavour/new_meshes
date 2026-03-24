import numpy as np
import sympy
import gmsh


def format_number(x):
    a, b = f'{x:.7e}'.split('e')
    if b[0] == '-':
        b = b[0] + b[2]
    else:
        b = b[2]
    return fr'{a} \cdot 10^{{{b}}}'


def read_npz(fname):
    data = np.load(fname)
    dofs_coords = data['dofs_coords']
    psi_w = data['psi_w']
    uv = data['uv']
    
    psi = psi_w[::2]
    w = psi_w[1::2]
    u = uv[::2]
    v = uv[1::2]
    return dofs_coords, psi, w, u, v


def run():
    fname = 'results.txt'
    data = np.loadtxt(fname)
    data = data[:, :-2]

    # n, p, psi_min, psi_max, w_min, w_max, time, mpi_size

    #print(r'% \renewcommand{\arraystretch}{1.5}')
    print(r'\begin{table}[h]')
    print(r'\caption{Эксперимент}')
    print(r'\begin{center}')
    print(r'\begin{tabular}' + '{' + f'{'|c' * 7 + '|'}' + '}')
    #'{' + f'{'|c' * 7 + '|'}' + '}'
    # '{|c|c|c|r|r|r|c|}'
    print(r'\hline')

    print(r'$m$ & $p$ & $\psi_{min}$ & $\psi_{max}$ & $\omega_{min}$ & $\omega_{max}$ \\')
    print(r'\hline')
    
    for i, row in enumerate(data, 1):
        n, p, psi_min, psi_max, w_min, w_max = row

        n, p = map(lambda x: str(int(x)), (n, p))
        psi_min = f'{psi_min:.8f}'
        psi_max, w_min, w_max = map(format_number, (psi_max, w_min, w_max))

        print(fr'$' + '$ & $'.join((n, p, psi_min, psi_max, w_min, w_max)) + r'$ \\')
    
    n = 800
    p = 2
    dofs_coords, psi, w, u, v = read_npz(f'data/NS/stokes_n{n}_p{p}.npz')

    n, p = map(lambda x: str(int(x)), (n, p))
    psi_min = f'{psi.min():.8f}'
    psi_max, w_min, w_max = map(format_number, (psi.max(), w.min(), w.max()))

    print(fr'$' + '$ & $'.join((n, p, psi_min, psi_max, w_min, w_max)) + r'$ \\')
    
    print(fr'--- & --- & $-0.10007627$ & ${sympy.latex(2.2276e-6)}$ & --- & --- \\')

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\label{table}')
    print(r'\end{table}')


def get_number_of_nodes_and_cells(mesh_name):
    gmsh.initialize()
    gmsh.open(mesh_name)
    node_tags, coords, _ = gmsh.model.mesh.get_nodes()
    element_tags, element_node_tags = gmsh.model.mesh.get_elements_by_type(gmsh.model.mesh.get_element_type("Triangle", 1))
    gmsh.finalize()
    return node_tags.size, element_tags.size


def run2():
    fname = 'results2.txt'
    data = np.genfromtxt(fname, usecols=(1, 2, 3, 4, 5))

    mesh_names = (
        'meshes/unit_square/unit_square_128.msh',
        'meshes/unit_square/unit_square_256.msh',
        'meshes/unit_square/unit_square_512.msh'
    )

    mesh_data = np.array([get_number_of_nodes_and_cells(mesh_name) * 3 for mesh_name in mesh_names]).flatten().reshape(-1, 2)

    # mesh_name, p, psi_min, psi_max, w_min, w_max, time, mpi_size
    # mesh_name -> nodes, cells

    print(r'% \renewcommand{\arraystretch}{1.5}')
    print(r'\begin{table}[h]')
    print(r'\caption{Эксперимент}')
    print(r'\begin{center}')
    print(r'\begin{tabular}' + '{' + f'{'|c' * 7 + '|'}' + '}')
    print(r'\hline')

    print(r'$nodes$ & $cells$ & $p$ & $\psi_{min}$ & $\psi_{max}$ & $\omega_{min}$ & $\omega_{max}$ \\')
    print(r'\hline')

    #data = data[:, :-2]

    for row, d in zip(data, mesh_data):
        p, psi_min, psi_max, w_min, w_max = row

        nodes, cells = d

        p = int(p)
        psi_min = sympy.N(psi_min, 8)
        psi_max = sympy.N(psi_max, 5)
        w_min = sympy.Float(w_min, 5)
        w_max = sympy.Float(w_max, 5)

        strs = map(sympy.latex, (nodes, cells, p, psi_min, psi_max, w_min, w_max))

        print(r'$' + '$ & $'.join(strs) + r'$ \\')

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\label{table}')
    print(r'\end{table}')


if __name__ == '__main__':
    run()


    asd = r'''
    \begin{center}
    \begin{tabular}{ |c|c|c| } 
        \hline
        cell1 & cell2 & cell3 \\ 
        cell4 & cell5 & cell6 \\ 
        cell7 & cell8 & cell9 \\ 
        \hline
    \end{tabular}
    \end{center}
    '''