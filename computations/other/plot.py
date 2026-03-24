import numpy as np
import matplotlib.pyplot as plt

n = 1000

name = f'experiment_triangle_{n}'
title = f'Сетка ({n} на {n}) * 2'

direct = []
for i in range(1, 7):
    direct.append(np.load(f'data/parallel/direct_{i}.npy'))

direct_with_options = []
for i in range(1, 7):
    direct_with_options.append(np.load(f'data/parallel/direct_{i}_with_options.npy'))

krylov = []
for i in range(1, 7):
    krylov.append(np.load(f'data/parallel/krylov_{i}.npy'))

krylov_with_options = []
for i in range(1, 7):
    krylov_with_options.append(np.load(f'data/parallel/krylov_{i}_with_options.npy'))

data = np.array((direct, direct_with_options, krylov, krylov_with_options))


f, axes = plt.subplots(2, 3, figsize=(6.4 * 2, 3.6 * 2), dpi=300, tight_layout=True)
f.suptitle(title)

# time
x = tuple(range(1, 7))

axes[0, 0].plot(x, data[:, :, 0].T)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xlabel('Кол-во процессов')
axes[0, 0].set_ylabel('Секунды')

axes[0, 0].grid()
axes[0, 0].legend(('Прямой', 'Прямой с опциями', 'Итерационный', 'Итерационный с опциями'))

#f.savefig(f'images/parallel/{name}_time.pdf', transparent=True)

# ускорение
x = tuple(range(1, 7))

axes[0, 1].plot(x, data[0, 0, 0] / data[0, :, 0])
axes[0, 1].plot(x, data[1, 0, 0] / data[1, :, 0])
axes[0, 1].plot(x, data[2, 0, 0] / data[2, :, 0])
axes[0, 1].plot(x, data[3, 0, 0] / data[3, :, 0])

axes[0, 1].set_xticks(x)
axes[0, 1].set_xlabel('Кол-во процессов')
axes[0, 1].set_ylabel('Ускорение')

axes[0, 1].grid()
axes[0, 1].legend(('Прямой', 'Прямой с опциями', 'Итерационный', 'Итерационный с опциями'))

#f.savefig(f'images/parallel/{name}_speedup.pdf', transparent=True)

# Эффективность
x = np.array(tuple(range(1, 7)))

axes[0, 2].plot(x, data[0, 0, 0] / data[0, :, 0] / x)
axes[0, 2].plot(x, data[1, 0, 0] / data[1, :, 0] / x)
axes[0, 2].plot(x, data[2, 0, 0] / data[2, :, 0] / x)
axes[0, 2].plot(x, data[3, 0, 0] / data[3, :, 0] / x)

axes[0, 2].set_xticks(x)
axes[0, 2].set_xlabel('Кол-во процессов')
axes[0, 2].set_ylabel('Эффективность')

axes[0, 2].grid()
axes[0, 2].legend(('Прямой', 'Прямой с опциями', 'Итерационный', 'Итерационный с опциями'))

#f.savefig(f'images/parallel/{name}_efficiency.pdf', transparent=True)

# Ошибки
for num, (i, error) in enumerate(zip(range(1, 4), ('$L^2$', r'$L^{\infty}$', '$H_0^1$'))):

    x = tuple(range(1, 7))

    axes[1, num].plot(x, data[:, :, i].T)

    axes[1, num].set_xticks(x)
    axes[1, num].set_xlabel('Кол-во процессов')
    axes[1, num].set_ylabel(error)

    axes[1, num].grid()
    axes[1, num].legend(('Прямой', 'Прямой с опциями', 'Итерационный', 'Итерационный с опциями'))

plt.savefig(f'images/parallel/{name}.pdf', transparent=True)