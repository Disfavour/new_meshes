import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

rect = Rectangle((-2,-2),4,2, facecolor="none", edgecolor="none")
circle = Circle((0,0),1)

ax = plt.axes()
ax.add_artist(rect)
ax.add_artist(circle)

circle.set_clip_path(rect)

plt.axis('equal')
plt.axis((-2,2,-2,2))
plt.show()