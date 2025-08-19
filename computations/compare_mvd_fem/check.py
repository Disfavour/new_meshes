import elliptic_fem
import elliptic_mvd
import numpy as np

if __name__ == '__main__':
    k = np.array(
        ((1, 0),
         (0, 1))
    )
    for i in range(10):
        res_mvd = elliptic_mvd.calculate(f'meshes/rectangle/old_rectangle_{i}_quadrangle', k)
        res_fem = elliptic_fem.solve(f'meshes/rectangle/rectangle_{i}_triangle.msh', k)

        u_D, node_coords, A_D, f_D, node_groups = res_mvd
        u, dof_coords, A, b1, b2, b3 = res_fem

        indexes = []
        for coord in node_coords:
            for i, coord2 in enumerate(dof_coords):
                if np.allclose(coord, coord2):
                    indexes.append(i)
                    break

        A = A[indexes]
        A = A[:, indexes]

        #print(node_groups)
        np.savetxt('test_mvd.txt', A_D, fmt='%+0.2f')
        np.savetxt('test_fem.txt', A, fmt='%+0.2f')

        assert np.allclose(A[:node_groups[0], :node_groups[0]], A_D[:node_groups[0], :node_groups[0]])
        assert np.allclose(A[node_groups[0]:, node_groups[0]:], A_D[node_groups[0]:, node_groups[0]:])

        b1 = b1[indexes]    # стартовый
        b2 = b2[indexes]    # после лифтинга
        b3 = b3[indexes]    # с граничными дирихле

        A_dbc = A_D[:node_groups[0], node_groups[0]:node_groups[1]]

        # предполагаем что правая верхняя подматрица у нас совпадает и делаем лифтинг

        b_inner_lifting = b1[:node_groups[0]] - A_dbc @ f_D[node_groups[0]:node_groups[1]]

        # print(b_inner_lifting)
        # print(b2)
        # print()
        assert np.allclose(b_inner_lifting, b2[:node_groups[0]])

        b_after_dirichlet = np.concatenate((b_inner_lifting, f_D[node_groups[0]:]))

        # print(b_after_dirichlet)
        # print(b3)
        # print()
        assert np.allclose(b_after_dirichlet, b3)
        
        # Векторы f для внутренних узлов немного отличаются (в мкэ квадратуры) -> u тоже немного отличается
        # Матрица полностью соответствует
