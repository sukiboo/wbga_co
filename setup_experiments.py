'''
    This file contains a class that sets up a sparse convex minimization problem
    and solves it various optimization algorithms
'''

import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import gmean
from algorithms import RWRGA, WGAFR, WCGA, L1REG

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='Set2', font='monospace', font_scale=1.5)


class SetupExperiments:
    '''setup and run the experiments'''

    def __init__(self, ex=1, seed=0):
        self.ex = ex
        self.seed = seed
        os.makedirs('./images/', exist_ok=True)

    def generate_dictionary(self, dim, num_d, type_d='gauss'):
        '''generate random dictionary'''
        if type_d == 'gauss':
            self.D = np.random.randn(num_d, dim)
        elif type_d == 'uniform':
            self.D = 2 * np.random.rand(num_d, dim) - 1
        self.p = 1 + 9*np.random.rand()
        self.D /= np.linalg.norm(self.D, self.p, axis=1, keepdims=True)

    def generate_element(self, min_sparsity, max_sparsity):
        '''generate element of random sparsity'''
        f_spr = np.random.randint(min_sparsity, max_sparsity)
        f_ind = np.random.randint(self.D.shape[0], size=f_spr)
        f_coef = np.random.randn(f_spr)
        noise = .1 * np.random.rand(self.D.shape[-1])
        f = np.matmul(f_coef, self.D[f_ind]) + noise
        return f

    def generate_function(self):
        '''generate objective function for each numerical example'''
        if self.ex == 1:
            # example 1
            self.max_sparsity = 100
            self.generate_dictionary(dim=1000, num_d=10000)
            f = self.generate_element(self.max_sparsity, 3*self.max_sparsity)
            q = 1 + 9*np.random.rand()
            E = lambda x: norm(x - f, q)**q
            dE = lambda x,y: q * norm(x - f, q)**(q-1) * self.F_lp(x - f, y, q)

        elif self.ex == 2:
            # example 2
            self.max_sparsity = 100
            self.generate_dictionary(dim=1000, num_d=10000)
            f = self.generate_element(2*self.max_sparsity, 4*self.max_sparsity)
            g = self.generate_element(2*self.max_sparsity, 4*self.max_sparsity)
            q = 2 + 8*np.random.rand()
            r = 1 + np.random.rand()
            E = lambda x: norm(x - f, q)**q + norm(x - g, r)**r
            dE = lambda x,y: q * norm(x - f, q)**(q-1) * self.F_lp(x - f, y, q)\
                           + r * norm(x - g, r)**(r-1) * self.F_lp(x - g, y, r)

        elif self.ex == 3:
            # example 3
            self.max_sparsity = 50
            self.generate_dictionary(dim=100, num_d=500)
            f = self.generate_element(self.max_sparsity, 2*self.max_sparsity)
            q = 1 + 9*np.random.rand()
            E = lambda x: norm(x - f, q)**q
            dE = lambda x,y: q * norm(x - f, q)**(q-1) * self.F_lp(x - f, y, q)

        elif self.ex == 4:
            # example 4
            self.max_sparsity = 50
            self.generate_dictionary(dim=100, num_d=500)
            f = self.generate_element(2*self.max_sparsity, 4*self.max_sparsity)
            g = self.generate_element(2*self.max_sparsity, 4*self.max_sparsity)
            q = 2 + 8*np.random.rand()
            r = 1 + np.random.rand()
            E = lambda x: norm(x - f, q)**q + norm(x - g, r)**r
            dE = lambda x,y: q * norm(x - f, q)**(q-1) * self.F_lp(x - f, y, q)\
                           + r * norm(x - g, r)**(r-1) * self.F_lp(x - g, y, r)

        else:
            raise SystemExit(f'\nexample {self.ex} is not implemented, available examples: 1,2,3,4\n')
        self.normalize_function(E, dE, f)

    def F_lp(self, x, y, p):
        '''norming functional in l_p-space'''
        F = np.matmul(y.reshape((-1,x.size)), (np.sign(x) * np.abs(x)**(p-1)).reshape((-1,1)))
        F /= norm(x,p)**(p-1) + 1e-08
        return F.ravel()

    def normalize_function(self, E, dE, x0):
        '''rescale the objective function to take values in [0,1]'''
        E0 = E(0)
        JE = lambda x: dE(x, np.eye(self.D.shape[-1]))
        E_min = minimize(E, x0=x0, jac=JE, tol=1e-06, options={'maxiter': 100000}).fun
        self.E = lambda x: (E(x) - E_min) / (E0 - E_min)
        self.dE = lambda x,y: dE(x,y) / (E0 - E_min)

    def run(self, num_exp):
        '''iteratively run experiments'''
        self.num_exp = num_exp
        self.err = {'wcga': {}, 'wgafr': {}, 'rwrga': {}, 'l1-reg': {}}
        self.ind = {'wcga': {}, 'wgafr': {}, 'rwrga': {}, 'l1-reg': {}}
        np.random.seed(self.seed)
        test_seeds = np.random.randint(1e+09, size=self.num_exp)
        # run experiments
        for t in range(self.num_exp):
            np.random.seed(test_seeds[t])
            self.generate_function()
            print(f'Running experiment {t+1}/{self.num_exp}:')
            # minimize the target function via various algorithms
            self.err['wcga'][t], self.ind['wcga'][t] = WCGA(self.E, self.dE, self.D, self.max_sparsity)
            self.err['wgafr'][t], self.ind['wgafr'][t] = WGAFR(self.E, self.dE, self.D, self.max_sparsity)
            self.err['rwrga'][t], self.ind['rwrga'][t] = RWRGA(self.E, self.dE, self.D, self.max_sparsity)
            if self.ex in [3,4]:
                self.err['l1-reg'][t], self.ind['l1-reg'][t] = L1REG(self.E, self.dE, self.D, self.max_sparsity)

    def plot_error_sparsity(self, perc=(25,75)):
        '''plot approximation errors vs the approximant sparsity'''
        fig, ax = plt.subplots(figsize=(8,6))
        for alg in self.err.keys():
            try:
                # process approximation data
                val = self.process_data(self.err[alg], self.ind[alg])
                val_avg = gmean(val, axis=0)
                val_min = np.percentile(val, perc[0], axis=0)
                val_max = np.percentile(val, perc[1], axis=0)
                # plot approximation errors
                ax.plot(np.arange(self.max_sparsity+1), val_avg, linewidth=3, label=alg)
                ax.fill_between(np.arange(self.max_sparsity+1), val_min, val_max, alpha=.25)
            except:
                pass
        # configure axes
        ax.set_yscale('log')
        ax.set_xlabel('solution sparsity')
        ax.set_ylabel('function value')
        ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(f'./images/ex{self.ex}_err_vs_spr.pdf', format='pdf')
        plt.show()

    def plot_error_iterations(self, i=None, perc=(25,75)):
        '''plot approximation errors vs the algorithm iterations -- the most distinct experiment'''
        fig, ax = plt.subplots(figsize=(8,6))
        if i is None:
            i = np.argmax([np.abs(self.err['rwrga'][t][-1]\
                                  / self.err['wgafr'][t][-1]) for t in range(self.num_exp)])
        for alg in self.err.keys():
            try:
                ax.plot(np.arange(len(self.err[alg][i])), self.err[alg][i], linewidth=4, label=alg)
            except:
                pass
        # configure axes
        ax.set_yscale('log')
        ax.set_xlabel('algorithm iterations')
        ax.set_ylabel('function value')
        ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(f'./images/ex{self.ex}_err_vs_itr.pdf', format='pdf')
        plt.show()

    def process_data(self, err, ind):
        '''process error-iteration data into error-sparsity data'''
        val = np.zeros((self.num_exp,self.max_sparsity+1))
        for t in range(self.num_exp):
            inds = []
            val[t][0] = 1
            for i in range(len(ind[t])):
                if ind[t][i] not in inds:
                    inds.append(ind[t][i])
                val[t][len(inds):] = err[t][i+1]
        return val

