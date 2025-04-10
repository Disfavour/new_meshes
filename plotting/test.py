import matplotlib.pyplot as plt


def arrow(p1, p2, **props):
    ax.annotate(
        "", p1, p2,  xycoords='figure fraction',
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, **props))


fig, ax = plt.subplots(figsize=(6.5, 4))
fig.set_facecolor('lightblue')
fig.subplots_adjust(0, 0, 1, 1, 0.4, 0.4)
#for ax in axs.flat:
#ax.set(xticks=[], yticks=[])


ax.margins(x=0, y=0)#default 0.05
#ax.set_axis_off()
#fig.tight_layout()
ax.plot((1, 2, 3), (1, 2, 3))

arrow((0, 0.75), (0.1, 0.75))  # left
arrow((0.435, 0.75), (0.565, 0.75))  # wspace
arrow((0.9, 0.75), (1, 0.75))  # right
fig.text(0.05, 0.7, "left", ha="center")
fig.text(0.5, 0.7, "wspace", ha="center")
fig.text(0.95, 0.7, "right", ha="center")

arrow((0.25, 0), (0.25, 0.1))  # bottom
arrow((0.25, 0.435), (0.25, 0.565))  # hspace
arrow((0.25, 0.9), (0.25, 1))  # top
fig.text(0.28, 0.05, "bottom", va="center")
fig.text(0.28, 0.5, "hspace", va="center")
fig.text(0.28, 0.95, "top", va="center")

plt.show()