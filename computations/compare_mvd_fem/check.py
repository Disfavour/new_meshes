import elliptic_fem
import elliptic_mvd
import numpy as np

if __name__ == '__main__':
    k = np.array(
        ((1, 0),
         (0, 1))
    )
    res_mvd = elliptic_mvd.calculate('meshes/rectangle/rectangle_0_quadrangle', k)
    res_fem = elliptic_fem.solve('meshes/rectangle/rectangle_0_triangle.msh', k)

    u_D, node_coords, A_D, f_D, node_groups = res_mvd
    u, dof_coords, A, b1, b2, b3 = res_fem

    indexes = []
    for coord in node_coords:
        for i, coord2 in enumerate(dof_coords):
            if np.allclose(coord, coord2):
                indexes.append(i)
                break
    
    print(indexes)
    print(node_coords)
    print(dof_coords[indexes])

    #print(A_D)
    A = A[indexes]
    A = A[:, indexes]
    #print(A)
    #print(A.shape)

    np.savetxt('test_mvd.txt', A_D, fmt='%+0.2f')
    np.savetxt('test_fem.txt', A, fmt='%+0.2f')

    b1 = b1[indexes]
    b2 = b2[indexes]
    b3 = b3[indexes]

    A_dbc = A_D[:node_groups[0], node_groups[0]:node_groups[1]]
    
    print(b1[:node_groups[0]] - A_dbc @ f_D[node_groups[0]:node_groups[1]])
    print(b2[:node_groups[0]])
    print()

    print(b3[node_groups[0]:node_groups[1]])
    print(f_D[node_groups[0]:node_groups[1]])
    print()

    print(b1[:node_groups[0]])
    print(f_D[:node_groups[0]])


    # print(np.sort(u_D))
    # print(np.sort(u))
