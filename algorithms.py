'''
    This file contains various algorithms that obtain a sparse (with respect to a given dictionary D)
    solution to the problem of minimizing a convex target function E
'''

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


def RWRGA(E, dE, D, max_sparsity):
    '''Rescaled Weak Relaxed Greedy Algorithm'''
    x = np.zeros(D.shape[-1])
    err, ind, spr = [E(0)], [], 0

    # iteratively construct the minimizer
    desc = lambda: f'  RWRGA(co): sparsity={spr:3d}/{max_sparsity}, value={err[-1]:.2e}'
    with tqdm(total=max_sparsity, ascii=True, desc=desc(), bar_format='{desc} |{bar}|') as pbar:
        itr = 0
        while (spr < max_sparsity) and (itr < 2*max_sparsity):

            # select element of the dictionary
            ind_max = np.argmax(np.abs(dE(x,D)))

            # update the minimizer
            la = minimize(lambda c: E(x + c*D[ind_max]), x0=0, tol=1e-06).x
            mu = minimize(lambda c: E(c*(x + la*D[ind_max])), x0=1, tol=1e-06).x
            x = mu * (x + la*D[ind_max])

            # update variables
            if ind_max not in ind:
                spr += 1
                pbar.update()
            ind.append(ind_max)
            err.append(E(x))
            pbar.desc = desc()
            itr += 1

    return err, ind


def WGAFR(E, dE, D, max_sparsity):
    '''Weak Greedy Algorithm with Free Relaxation'''
    x = np.zeros(D.shape[-1])
    err, ind, spr = [E(0)], [], 0

    # iteratively construct the minimizer
    desc = lambda: f'  WGAFR(co): sparsity={spr:3d}/{max_sparsity}, value={err[-1]:.2e}'
    with tqdm(total=max_sparsity, ascii=True, desc=desc(), bar_format='{desc} |{bar}|') as pbar:
        itr = 0
        while (spr < max_sparsity) and (itr < 2*max_sparsity):

            # select element of the dictionary
            ind_max = np.argmax(np.abs(dE(x,D)))

            # update the minimizer
            w, la = minimize(lambda c: E((1-c[0])*x + c[1]*D[ind_max]), x0=[0,0], tol=1e-06).x
            x = (1 - w)*x + la*D[ind_max]

            # update variables
            if ind_max not in ind:
                spr += 1
                pbar.update()
            ind.append(ind_max)
            err.append(E(x))
            pbar.desc = desc()
            itr += 1

    return err, ind


def WCGA(E, dE, D, max_sparsity):
    '''Weak Chebyshev Greedy Algorithm'''
    x = np.zeros(D.shape[-1])
    err, ind, spr, coef = [E(0)], [], 0, []

    # iteratively construct the minimizer
    desc = lambda: f'   WCGA(co): sparsity={spr:3d}/{max_sparsity}, value={err[-1]:.2e}'
    with tqdm(total=max_sparsity, ascii=True, desc=desc(), bar_format='{desc} |{bar}|') as pbar:
        itr = 0
        while (spr < max_sparsity) and (itr < max_sparsity):

            # select element of the dictionary
            ind_max = np.argmax(np.abs(dE(x,D)))
            ind.append(ind_max)
            spr += 1

            # update the minimizer
            coef = minimize(lambda c: E(np.matmul(c, D[ind])), x0=[*coef,0], tol=1e-06).x
            x = np.matmul(coef, D[ind])

            # update variables
            err.append(E(x))
            pbar.update()
            pbar.desc = desc()
            itr += 1

    return err, ind


def L1REG(E, dE, D, max_sparsity, reg_coef=.1, reg_rate=.95):
    '''l1-regularized optimization'''
    err, ind, spr, coef = [E(0)], [], [0], np.zeros(D.shape[0])

    # solve minimzation problem with various regularization constants
    desc = lambda: f'     l1-reg: sparsity={spr[-1]:3d}/{max_sparsity}, value={err[-1]:.2e}'
    with tqdm(total=max_sparsity, ascii=True, desc=desc(), bar_format='{desc} |{bar}|') as pbar:
        while spr[-1] < max_sparsity:

            # solve l1-regularized problem
            E_reg = lambda x: E(np.matmul(x,D)) + reg_coef*np.sum(np.abs(x))
            res = minimize(E_reg, x0=coef, tol=1e-05, options={'maxiter': 10000}, method='Powell')
            coef = res.x
            if not res.success:
                print(f'l1-minimization failed: {res.message}')

            # update variables
            spr.append(np.sum(np.abs(coef) > np.amax(np.abs(coef)) / 100))
            err.append(E(np.matmul(coef,D)))
            pbar.desc = desc()
            pbar.update(spr[-1] - spr[-2])

            # decrease regularization constant
            reg_coef *= reg_rate

    # linearly interpolate optimization results
    ind = np.arange(max_sparsity+1)
    err = np.interp(ind, spr, err)

    return err, ind[1:]

