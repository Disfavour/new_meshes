import streamlit as st
import pandas as pd
import os
import os.path
import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd


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

qualities = {
    'minDetJac':    'the adaptively computed minimal Jacobian determinant',
    'maxDetJac':    'the adaptively computed maximal Jacobian determinant',
    'minSJ':        'sampled minimal scaled jacobien',
    'minSICN':      'sampled minimal signed inverted condition number',
    'minSIGE':      'sampled signed inverted gradient error',
    'gamma':        'ratio of the inscribed to circumcribed sphere radius (радиус вписанной к радиусу описанной, нормированный на единицу)',
    'innerRadius':  'радиус вписанное окружности',
    'outerRadius':  'радиус описанной окружности',
    'minIsotropy':  'minimum isotropy measure',
    'angleShape':   'angle shape measure',
    'minEdge':      'minimum straight edge length',
    'maxEdge':      'maximum straight edge length',
    'volume':       'площадь',
    'min_angle':    'минимальные углы',
    'max_angle':    'Максимальные углы',
    #'angles':       'Гистограмма всех углов',
}

with st.sidebar:
    '#### Набор сеток'
    number_of_set = st.slider('num of set', 1, number_of_sets, label_visibility='collapsed')
    name = f'{name}_{number_of_set}'

    '#### Качество'
    quality = st.selectbox('Качество', qualities, 14, label_visibility='collapsed')
    qualities[quality]

    '#### Размер точек (узлов)'
    msize = st.slider('Размер точек (узлов)', 0, 20, 5, label_visibility='collapsed')

    '#### Кол-во столбцов в гистограмме'
    number_of_bins = st.slider('num of bins', 1, 30, 10, label_visibility='collapsed')

    '#### Сетки на гистограмме'
    mesh_options = (
        'Базовая',
        '4-угольная',
        '4-угольная 2',
        'Перпендикуляры',
        'Медианы',
        'Биссектрисы',
        'Высоты',
        'split',
    )
    selection = st.multiselect("Сетки на графиках", mesh_options, mesh_options, label_visibility='collapsed')
    selection_idxs = [mesh_options.index(i) for i in selection]

mesh_names = [
        f'{name}_triangle',
        f'{name}_quadrangle',
        f'{name}_small_quadrangle',
        f'{name}_circumcenter',
        f'{name}_centroid',
        f'{name}_incenter',
        f'{name}_orthocenter',
        f'{name}_split_quadrangles'
    ]
number_of_different_meshes = len(mesh_names)
descriptions = (
    '#### Базовая',
    '#### Четырехугольная',
    '#### Четырехугольная 2',
    '#### Пересечение перпендикуляров',
    '#### Пересечение медиан',
    '#### Пересечение биссектрис',
    '#### Пересечение высот',
    '#### Четырехугольная split',
)

#mesh_names_msh = [os.path.join(meshes_msh_dir, f'{mesh_name}.msh') for mesh_name in mesh_names]
mesh_names_xdmf = [os.path.join(meshes_xdmf_dir, f'{mesh_name}.xdmf') for mesh_name in mesh_names]
mesh_names_npz = [os.path.join(data_dir, f'{mesh_name}.npz') for mesh_name in mesh_names]

def draw_mesh(mesh, array=None, vmin=None, vmax=None):
    cells = mesh.cells[0].data
    nodes = mesh.points
    
    verts = [
        [nodes[node] for node in cell] for cell in cells
    ]
    
    pc = PolyCollection(verts, facecolors='white', edgecolors='black')
    if array is not None:
        pc.set_array(array)
        pc.set_clim(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(6.4, 3.6), dpi=300, tight_layout=True)
    ax = plt.gca()
    if array is None:
        ax.add_collection(pc)
    else:
        plt.colorbar(ax.add_collection(pc))
    
    plt.plot(np.array(verts)[:, :, 0], np.array(verts)[:, :, 1], 'ko', markersize=msize)

    ax.autoscale()
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    return plt.gcf()

def draw_hist(d, bins):
    plt.figure(figsize=(6.4, 3.6), dpi=300, tight_layout=True)
    number_of_sets, bins, patches = plt.hist(d, bins)
    plt.xticks(bins, rotation='vertical')
    plt.grid()
    return plt.gcf()

complete = 0.0
progress_text = 'Загрузка'
progress_bar = st.progress(complete, text=progress_text)

meshes = []
data = []
dataframes = []
#base_plots = []
for xdmf, npz in zip(mesh_names_xdmf, mesh_names_npz):
    meshes.append(meshio.read(xdmf))
    data.append(np.load(npz))

    nodes = meshes[-1].points.shape[0]
    cells = meshes[-1].cells[0].data.shape[0]
    min_angle = data[-1]['min_angle'].min()
    max_angle = data[-1]['max_angle'].max()

    dataframes.append(
        pd.DataFrame([[nodes, cells, min_angle, max_angle]],
                    columns=['Узлы', 'Ячейки', 'Min угол', 'Max угол'])
    )
    
    #base_plots.append(draw_mesh(meshes[-1]))

    complete += 0.5
    progress_bar.progress(complete / number_of_sets, text=progress_text)

#progress_bar.empty()

min_quality, max_quality = data[0][quality].min(), data[0][quality].max()
for d in data[1:]:
    min_quality = min(min_quality, d[quality].min())
    max_quality = max(max_quality, d[quality].max())
#bins = np.linspace(min_quality, max_quality, number_of_bins + 1)

#hists = []
plots = []
data_hist = []
for d, mesh in zip(data, meshes):
    #hists.append(draw_hist(d[quality], bins))
    plots.append(draw_mesh(mesh, d[quality], min_quality, max_quality))

    data_hist.append(d[quality])

    complete += 0.5
    progress_bar.progress(complete / number_of_different_meshes, text=progress_text)

progress_bar.empty()

# print(selection_idxs, np.array(selection_idxs), np.array(selection_idxs).dtype)
# print(data_hist)
# data_hist = np.array(data_hist)
# data_hist = data_hist[np.array(selection_idxs)]

data_hist = [data_hist[i] for i in selection_idxs]

min_quality2, max_quality2 = data_hist[0].min(), data_hist[0].max()
for d in data_hist[1:]:
    min_quality2 = min(min_quality2, d.min())
    max_quality2 = max(max_quality2, d.max())
bins = np.linspace(min_quality2, max_quality2, number_of_bins + 1)


k = number_of_different_meshes // 2
for cur_descriptions, cur_plots, cur_dfs in zip((descriptions[:k], descriptions[k:]), (plots[:k], plots[k:]), (dataframes[:k], dataframes[k:])):
    columns = st.columns(len(cur_plots))
    for i, (column, cur_description, cur_plot, cur_df) in enumerate(zip(columns, cur_descriptions, cur_plots, cur_dfs)):
        with column:
            #st.pyplot(base_plots[i])
            cur_description
            st.pyplot(cur_plot)
            st.dataframe(cur_df, hide_index=True)
            
            #st.pyplot(hists[i])


plt.figure(figsize=(1.5*6.4, 1.5*3.6), dpi=300, tight_layout=True)
number_of_sets, bins, patches = plt.hist(data_hist, bins)
plt.xticks(bins, rotation='vertical')
plt.grid()
plt.legend(selection)

with st.columns((0.25, 0.5, 0.25))[1]:
    st.pyplot(plt.gcf())
