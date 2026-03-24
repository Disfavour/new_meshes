from multiprocessing import Pool
import method_3
import method_4
import time


def run():
    mesh_fname = 'meshes/NS/ellipse_256.msh'
    #Res = [10000] # 5500, 10000
    #sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # kappas = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    # kappas = [0.5]

    Re = 10000
    #sigmas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    sigmas = [0.05, 0.1, 0.15, 0.2]
    kappas = [0.0, 0.25, 0.5, 0.75, 1.0]

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    degree = 2
    max_iter = 500

    with Pool(6) as p:
        results = []
        results.append(p.starmap_async(method_3.solve, ((mesh_fname, Re, k, sigma, f'{prefix}_k{k}_s{sigma}', degree, max_iter) for k in kappas for sigma in sigmas)))
        results.append(p.starmap_async(method_4.solve, ((mesh_fname, Re, sigma, f'{prefix}_s{sigma}', degree, max_iter) for sigma in sigmas)))

        list(map(lambda x: x.wait(), results))
        # print(p._processes) # 12 default


def run2():
    mesh_fname = 'meshes/NS/ellipse_256.msh'
    Re = 10000
    
    kappa = 0.5
    sigmas1 = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]

    sigmas2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]

    degree = 2
    max_iter = 1000 # по идее хватит 500, но если на ночь, то можно и 1000. Надо 1000 чтобы они сошлись и картинки рисовать по данным.
    prefix = f'data/NS/ellipse_256_Re{Re}'
    with Pool(6) as p:
        results = []
        results.append(p.starmap_async(method_3.solve, ((mesh_fname, Re, kappa, sigma, f'{prefix}_k{kappa}_s{sigma}', degree, max_iter) for sigma in sigmas1)))
        results.append(p.starmap_async(method_4.solve, ((mesh_fname, Re, sigma, f'{prefix}_s{sigma}', degree, max_iter) for sigma in sigmas2)))

        list(map(lambda x: x.wait(), results))


def run3():
    mesh_fname = 'meshes/NS/ellipse_256.msh'
    Re = 10000
    
    kappa = 0.5
    #sigmas = [0.112, 0.114, 0.116, 0.118, 0.122, 0.124, 0.126, 0.128]
    sigmas = [0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129]

    degree = 2
    max_iter = 1000
    prefix = f'data/NS/ellipse_256_Re{Re}'
    with Pool(6) as p:
        res = p.starmap_async(method_3.solve, ((mesh_fname, Re, kappa, sigma, f'{prefix}_k{kappa}_s{sigma}', degree, max_iter) for sigma in sigmas))
        res.wait()


def run4():
    mesh_fname = 'meshes/NS/ellipse_256.msh'
    Re = 10000
    
    kappa = 0.5
    #sigmas = [0.112, 0.114, 0.116, 0.118, 0.122, 0.124, 0.126, 0.128]
    sigmas = [0.1155, 0.1165, 0.1175, 0.1185, 0.1195, 0.1205]

    degree = 2
    max_iter = 600
    prefix = f'data/NS/ellipse_256_Re{Re}'
    with Pool(6) as p:
        res = p.starmap_async(method_3.solve, ((mesh_fname, Re, kappa, sigma, f'{prefix}_k{kappa}_s{sigma}', degree, max_iter) for sigma in sigmas))
        res.wait()


if __name__ == '__main__':
    start_time = time.time()
    run4()
    print(time.time() - start_time)
