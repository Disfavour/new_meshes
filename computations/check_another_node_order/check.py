import elliptic_mvd_1
import elliptic_mvd_2
import numpy as np


k = np.array((
    (1, 0),
    (0, 1),
))

err1, u1, f1, A1 = elliptic_mvd_1.calculate('meshes/rectangle/old_rectangle_0_quadrangle', k)
err2, u2, f2, A2, u11 = elliptic_mvd_2.calculate('meshes/rectangle/rectangle_0_quadrangle', k)

print(np.array_equal(err1, err2), np.array_equal(u1, u2), np.array_equal(f1, f2), np.array_equal(A1, A2), np.array_equal(u11, u1))
print(np.allclose(err1, err2), np.allclose(u1, u2), np.allclose(f1, f2), np.allclose(A1, A2), np.allclose(u11, u1))

# При лифтинге немного меняется решение и, следовательно, нормы.

#np.savetxt('test_A1.txt', A1, fmt='%+0.2f')
#np.savetxt('test_A2.txt', A2, fmt='%+0.2f')

# print(err1)
# print(err2)

# print(u1)
# print(u2)

# print(f1)
# print(f2)
