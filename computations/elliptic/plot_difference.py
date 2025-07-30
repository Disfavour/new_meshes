import matplotlib.pyplot as plt
import matplotlib.tri as tri


def get_plot(node_coords_mvd, node_coords_fem, u_mvd, u_fem, u_exact, L_inf, triangle_node_tags, node_coords):

    triangulation = tri.Triangulation(node_coords_mvd[:, 0], node_coords_mvd[:, 0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)

    # Общий цветовой диапазон
    vmin = min(u_mvd.min(), u_fem.min(), u_exact.min())
    vmax = max(u_mvd.max(), u_fem.max(), u_exact.max())

    #fig.suptitle(fr'$L_\infty$ = {L_inf}')

    tpc0 = axes[0].tripcolor(triangulation, u_mvd, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[0].set_title("MVD")
    axes[0].axis('scaled')

    tpc1 = axes[1].tripcolor(node_coords_fem[:, 0], node_coords_fem[:, 0], u_fem, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[1].set_title("FEM")
    axes[1].axis('scaled')

    tpc2 = axes[2].tripcolor(triangulation, u_exact, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[2].set_title("Аналитическое решение")
    axes[21].axis('scaled')

    fig.colorbar(tpc2, ax=axes, orientation='vertical', label='u')

    #plt.show()

    return fig
    