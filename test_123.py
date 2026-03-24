import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi)
y = np.sin(x) * np.pi

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y)
ax1.axis('scaled')  # Масштабирует box

ax2.plot(x, y)
ax2.axis('equal')   # Корректирует лимиты
plt.show()