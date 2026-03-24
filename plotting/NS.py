import matplotlib.pyplot as plt
import sys
import numpy as np
import utility
import os.path


def plot(u, legend, fname, ymin=None, ymax=None, markers=True, loc='best'):
    fig, ax = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)

    line_type = 'o-' if markers else '-'

    #ax.plot(range(1, u.shape[1] + 1), u.T, line_type)
    assert u.shape[0] <= 20
    cmap = plt.get_cmap('tab10') if u.shape[0] <= 10 else plt.get_cmap('tab20')
    ncols = 1 if u.shape[0] <= 10 else 2
    x = range(1, u.shape[1] + 1)
    for i, v in enumerate(u):
        ax.plot(x, v, line_type, color=cmap(i))
    
    xticks = ax.get_xticks()[1:-1]
    xticks[0] = 1
    ax.set_xticks(xticks) # (1, 5, 10, 15, 20)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.grid()
    ax.legend(legend, loc=loc, ncols=ncols)

    ax.set_yscale('log')

    fig.savefig(fname, transparent=True)

    plt.close()


def plot_bisection(Res, errs, fname, ymin=None, ymax=None):
    fig, ax1 = plt.subplots(figsize=utility.get_default_figsize(), constrained_layout=True)

    color = 'tab:red'
    ax1.set_xlabel('$k$')
    ax1.set_ylabel(r'$\varepsilon$', color=color)
    ax1.plot(range(1, errs.size + 1), errs.flatten(), color=color)
    #ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.set_ylim(ymin, ymax)
    ax1.grid()

    xticks = ax1.get_xticks()[1:-1]
    xticks[0] = 1
    ax1.set_xticks(xticks) # (1, 100, 200, 300, 400, 500)

    new_Res = np.empty(Res.size * 2)
    new_Res[::2] = Res
    new_Res[1::2] = Res

    # (1, 10), (11, 20), (21, 30), ...
    k_Res = np.empty(Res.size * 2)
    k_Res[::2] = np.arange(1, 500, 10)
    k_Res[1::2] = np.arange(10, 501, 10)

    # print(k_Res)
    # print(new_Res)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(r'$\operatorname{Re}$', color=color)  # we already handled the x-label with ax1
    ax2.plot(k_Res, new_Res, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)
    #ax2.grid(color=color)

    fig.savefig(fname, transparent=True)
    plt.close()


def convergence_different_meshes():
    Res = [500, 525, 550, 575, 600, 625, 650]
    legend = [rf'$\operatorname{{Re}} = {Re}$' for Re in Res]
    ms = [64, 128, 256, 512]
    p = 2
    for m in ms:
        data = []
        for Re in Res:
            npz = np.load(f'data/NS/convergence_m{m}_p{p}_Re{Re}.npz')
            data.append(npz['errs_w'])

        plot(np.array(data), legend, f'images/NS/convergence_m{m}_p{p}.pdf', ymin=1e-12, ymax=1e9)


def convergence_different_p():
    Res = [500, 525, 550, 575, 600, 625, 650]
    legend = [rf'$\operatorname{{Re}} = {Re}$' for Re in Res]
    ms = [128, 512]
    ps = [1, 3]
    for m in ms:
        for p in ps:
            data = []
            for Re in Res:
                npz = np.load(f'data/NS/convergence_m{m}_p{p}_Re{Re}.npz')
                data.append(npz['errs_w'])

            plot(np.array(data), legend, f'images/NS/convergence_m{m}_p{p}.pdf', ymin=1e-12, ymax=1e9)


def convergence_different_forms():
    Res = [500, 525, 550, 575, 600, 625, 650]
    legend = [rf'$\operatorname{{Re}} = {Re}$' for Re in Res]
    forms = [2, 3]
    ms = [128, 512]
    p = 2
    for form in forms:
        for m in ms:
            data = []
            for Re in Res:
                npz = np.load(f'data/NS/convergence{form}_m{m}_p{p}_Re{Re}.npz')
                data.append(npz['errs_w'])

            plot(np.array(data), legend, f'images/NS/convergence_form{form}_m{m}_p{p}.pdf', ymin=1e-12, ymax=1e9)


def convergence_steps():
    ms = [128, 512]
    p = 2
    for m in ms:
        npz = np.load(f'data/NS/steps_m{m}_p{p}.npz')
        Res = npz['Res']
        errs = npz['errs']

        plot(errs, [rf'$\operatorname{{Re}} = {int(Re)}$' for Re in Res], f'images/NS/steps_m{m}_p{p}.pdf', ymin=1e-12, ymax=1e2)


def bisection():
    ms = [128, 512]
    p = 2
    for m in ms:
        npz = np.load(f'data/NS/bisection_m{m}_p{p}.npz')
        Res = npz['Res']
        errs = npz['errs']
        plot_bisection(Res, errs, f'images/NS/bisection_m{m}_p{p}.pdf', ymin=1e-12, ymax=1e9)


def iterative_methods():
    methods = [1, 2, 3, 4]
    m = 128
    p = 2
    Res = [1000] # , 5000, 10000
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sigmas.reverse()

    for Re in Res:
        for method in methods:
            data = []
            for sigma in sigmas:
                data.append(np.load(f'data/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}_s{sigma}.npz')['errs'][:200] / sigma)
            legend = [fr'$\sigma = {s}$' for s in sigmas]
            plot(np.array(data), legend, f'images/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')
    
    # forms = (2, 3)
    # for form in forms:
    #     for Re in Res:
    #         for method in methods:
    #             data = []
    #             for sigma in sigmas:
    #                 data.append(np.load(f'data/NS/form_{form}_iterative_method_{method}_m{m}_p{p}_Re{Re}_s{sigma}.npz')['errs'] / sigma)
    #             legend = [fr'$\sigma = {s}$' for s in sigmas]
    #             plot(np.array(data), legend, f'images/NS/form_{form}_iterative_method_{method}_m{m}_p{p}_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False)
    
    sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    sigmas.reverse()
    Re = 5000
    for method in methods:
            data = []
            for sigma in sigmas:
                data.append(np.load(f'data/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}_s{sigma}.npz')['errs'] / sigma)
            legend = [fr'$\sigma = {s}$' for s in sigmas]
            plot(np.array(data), legend, f'images/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')
    

    sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    sigmas.reverse()
    Re = 10000
    for method in methods:
        data = []
        for sigma in sigmas:
            data.append(np.load(f'data/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}_s{sigma}.npz')['errs'] / sigma)
        legend = [fr'$\sigma = {s}$' for s in sigmas]
        plot(np.array(data), legend, f'images/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')
    
    #sigmas = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    sigmas = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
    sigmas.reverse()
    Re = 20000
    for method in methods:
        data = []
        for sigma in sigmas:
            data.append(np.load(f'data/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}_s{sigma}.npz')['errs'] / sigma)
        legend = [fr'$\sigma = {s}$' for s in sigmas]
        plot(np.array(data), legend, f'images/NS/iterative_method_{method}_m{m}_p{p}_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_iterative_methods():
    Re = 5500
    kappas = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sigmas.reverse()
    
    for k in kappas:
        data = []
        for sigma in sigmas:
            fname_3 = f'data/NS/ellipse_128_Re{Re}_k{k}_s{sigma}'
            data.append(np.load(f'{fname_3}.npz')['errs'] / sigma)
        
        legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
        plot(np.array(data), legend, f'images/NS/ellipse_err_Re{Re}_k{k}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')
    
    data = []
    for sigma in sigmas:
        fname_4 = f'data/NS/ellipse_128_Re{Re}_s{sigma}'
        data.append(np.load(f'{fname_4}.npz')['errs'] / sigma)
    
    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
    plot(np.array(data), legend, f'images/NS/ellipse_err_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    Re = 10000
    k = 0.5

    data = []
    for sigma in sigmas:
        fname_3 = f'data/NS/ellipse_128_Re{Re}_k{k}_s{sigma}'
        data.append(np.load(f'{fname_3}.npz')['errs'] / sigma)
    
    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
    plot(np.array(data), legend, f'images/NS/ellipse_err_Re{Re}_k{k}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    data = []
    for sigma in sigmas:
        fname_4 = f'data/NS/ellipse_128_Re{Re}_s{sigma}'
        data.append(np.load(f'{fname_4}.npz')['errs'] / sigma)
    
    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
    plot(np.array(data), legend, f'images/NS/ellipse_err_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256():
    Re = 10000
    sigmas = [0.05, 0.1, 0.15, 0.2]
    kappas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sigmas.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]

    for k in kappas:
        data = []
        for sigma in sigmas:
            data.append(np.load(f'{prefix}_k{k}_s{sigma}.npz')['errs'][:500] / sigma)
        plot(np.array(data), legend, f'images/NS/ellipse_256_err_Re{Re}_k{k}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')
    
    data = []
    for sigma in sigmas:
        data.append(np.load(f'{prefix}_s{sigma}.npz')['errs'][:500] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_2():
    Re = 10000
    kappa = 0.5
    sigmas1 = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    # [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    # [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    sigmas1.reverse()

    sigmas2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    sigmas2.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas1]
    data = []
    for sigma in sigmas1:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err2_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas2]
    data = []
    for sigma in sigmas2:
        data.append(np.load(f'{prefix}_s{sigma}.npz')['errs'] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err2_Re{Re}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_3():
    Re = 10000
    kappa = 0.5
    sigmas = [0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129, 0.13]
    sigmas.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
    data = []
    for sigma in sigmas:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err3_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_4():
    Re = 10000
    kappa = 0.5
    sigmas1 = [0.119, 0.12, 0.121]
    sigmas2 = [0.116, 0.117, 0.118]
    sigmas1.reverse()
    sigmas2.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas1]
    data = []
    for sigma in sigmas1:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err4_1_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas2]
    data = []
    for sigma in sigmas2:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err4_2_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_5():
    Re = 10000
    kappa = 0.5
    sigmas = [0.1165, 0.117, 0.1175, 0.118, 0.1185, 0.119, 0.1195, 0.12, 0.1205, 0.121]
    sigmas.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas]
    data = []
    for sigma in sigmas:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'][:600] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err5_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_6():
    Re = 10000
    kappa = 0.5
    sigmas1 = [0.1185, 0.119, 0.1195, 0.12, 0.1205]
    sigmas2 = [0.1165, 0.117, 0.1175, 0.118]
    sigmas1.reverse()
    sigmas2.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_256_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas1]
    data = []
    for sigma in sigmas1:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'][:600] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err6_1_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas2]
    data = []
    for sigma in sigmas2:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'][:600] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_256_err6_2_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


def ellipse_256_7():
    Re = 10000
    kappa = 0.5
    sigmas1 = [0.1185, 0.119, 0.1195, 0.12, 0.1205]
    sigmas2 = [0.1165, 0.117, 0.1175, 0.118]
    sigmas1.reverse()
    sigmas2.reverse()

    #fname = f'data/NS/ellipse_256_Re{Re}_k{k}_s{sigma}'
    prefix = f'data/NS/ellipse_3_Re{Re}'

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas1]
    data = []
    for sigma in sigmas1:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'][:600] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_3_err7_1_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')

    legend = [fr'$\sigma = {sigma}$' for sigma in sigmas2]
    data = []
    for sigma in sigmas2:
        data.append(np.load(f'{prefix}_k{kappa}_s{sigma}.npz')['errs'][:600] / sigma)
    plot(np.array(data), legend, f'images/NS/ellipse_3_err7_2_Re{Re}_k{kappa}.pdf', ymin=1e-12, ymax=1e12, markers=False, loc='upper right')


if __name__ == '__main__':
    # convergence_different_meshes()
    # convergence_different_p()
    # convergence_different_forms()
    # convergence_steps()
    # bisection()
    # iterative_methods()
    # ellipse_iterative_methods()
    # ellipse_256()
    # ellipse_256_2()
    # ellipse_256_3()
    # ellipse_256_4()
    # ellipse_256_5()
    # ellipse_256_6()
    ellipse_256_7()
