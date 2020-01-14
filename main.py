'''
	this file contains the code for the numerical experiments presented in the paper
	'Biorthogonal greedy algorithms in convex optimization'
	the four examples presented in the paper can be recreated by running the code
	with an appropriate label ('ex1', 'ex2', 'ex3', and 'ex4' respectively)
'''

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from auxiliary_functions import *
from optimization_algorithms import wcga, wgafr, rwrga, regularized_optimization

# set print options
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})


''' run a simulation of a selected setting '''
def run_simulation(example, max_spr):

	if example == 'ex1':
		# parameters for random generation
		dim = 500
		num_dict = 1000
		f_spr = 60
		# generate dictionary and element
		D = random_dictionary(num_dict, dim)
		f = random_element(D, f_spr)
		# define target function
		p = 1.2
		E = lambda x: norm(x-f,p)**p
		dE = lambda x,y: p * norm(x-f,p)**(p-1) * F_lp(x-f,y,p)

	elif example == 'ex2':
		# parameters for random generation
		dim = 500
		num_dict = 1000
		f_spr, g_spr = 30, 30
		# generate dictionary and element
		D = random_dictionary(num_dict, dim)
		f, g = random_element(D, f_spr), random_element(D, g_spr)
		# define target function
		p, q = 3, 1.2
		E = lambda x: norm(x-f,p)**p * norm(g,q)**q + norm(x-g,q)**q * norm(f,p)**p
		dE = lambda x,y: p * norm(x-f,p)**(p-1) * F_lp(x-f,y,p) * norm(g,q)**q\
							+ q * norm(x-g,q)**(q-1) * F_lp(x-g,y,q) * norm(f,p)**p

	elif example == 'ex3':
		# parameters for random generation
		dim = 100
		num_dict = 200
		f_spr, g_spr = 30, 30
		# generate dictionary and element
		D = random_dictionary(num_dict, dim)
		f, g = random_element(D, f_spr), random_element(D, g_spr)
		# define target function
		p, q = 4, 1.5
		E = lambda x: norm(x-f,p)**p * norm(g,q)**q + norm(x-g,q)**q * norm(f,p)**p
		dE = lambda x,y: p * norm(x-f,p)**(p-1) * F_lp(x-f,y,p) * norm(g,q)**q\
							+ q * norm(x-g,q)**(q-1) * F_lp(x-g,y,q) * norm(f,p)**p

	elif example == 'ex4':
		# parameters for random generation
		dim = 200
		# generate dictionary and element
		D = canonical_dictionary(dim)
		f, g = random_element(D, dim), random_element(D, dim)
		# define target function
		p, q = 7, 3
		E = lambda x: norm(x-f,p)**p * norm(g,q)**q + norm(x-g,q)**q * norm(f,p)**p
		dE = lambda x,y: p * norm(x-f,p)**(p-1) * F_lp(x-f,y,p) * norm(g,q)**q\
							+ q * norm(x-g,q)**(q-1) * F_lp(x-g,y,q) * norm(f,p)**p

	# scale target function to be in [0,1]
	E0 = E(0)
	H = lambda x: E(x) / E0
	dH = lambda x,y: dE(x,y) / E0

	# optimization via non-sparse minimization
	H_min = minimize(H, f, tol=1e-04, options={'maxiter': 100000}).fun
	print('\nMinimial function value: {:.2e}'.format(H_min))

	# optimization via greedy algorithms
	x_rwrga, ind_rwrga, appr_rwrga = rwrga(H, dH, D, max_spr)
	x_wgafr, ind_wgafr, appr_wgafr = wgafr(H, dH, D, max_spr)
	x_wcga, ind_wcga, appr_wcga = wcga(H, dH, D, max_spr)

	# optimization via regularized minimization
	if example in ['ex3','ex4']:
		appr_reg = regularized_optimization(H, D, max_spr)
		appr_reg = (np.array(appr_reg) - H_min) / (1 - H_min)
	else:
		appr_reg = np.zeros(max_spr+1)

	# adjust the optimization results
	appr_rwrga = (np.array(appr_rwrga) - H_min) / (1 - H_min)
	appr_wgafr = (np.array(appr_wgafr) - H_min) / (1 - H_min)
	appr_wcga = (np.array(appr_wcga) - H_min) / (1 - H_min)

	return appr_rwrga, len(ind_rwrga), appr_wgafr, len(ind_wgafr), appr_wcga, len(ind_wcga), appr_reg



''' select a setting and a number of simulations to run '''
# specify the example, number of simulations, and the maximal sparsity
example = 'ex1'
num_sim = 10
max_spr = 50

# initialize variables
appr_rwrga, itr_rwrga = np.zeros((num_sim,max_spr+1)), np.zeros(num_sim)
appr_wgafr, itr_wgafr = np.zeros((num_sim,max_spr+1)), np.zeros(num_sim)
appr_wcga, itr_wcga = np.zeros((num_sim,max_spr+1)), np.zeros(num_sim)
appr_reg = np.zeros((num_sim,max_spr+1))

# run simulations
for i in range(num_sim):
	print('\n====== Running simulation {:d} / {:d} ======'.format(i+1, num_sim))
	appr_rwrga[i], itr_rwrga[i], appr_wgafr[i], itr_wgafr[i], \
		appr_wcga[i], itr_wcga[i], appr_reg[i] = run_simulation(example, max_spr)

# calculate statistical distribution of optimization results
stat_rwrga = np.concatenate((\
				np.amax(appr_rwrga, axis=0, keepdims=True),\
				np.mean(appr_rwrga, axis=0, keepdims=True),\
				np.amin(appr_rwrga, axis=0, keepdims=True)), axis=0)
stat_wgafr = np.concatenate((\
				np.amax(appr_wgafr, axis=0, keepdims=True),\
				np.mean(appr_wgafr, axis=0, keepdims=True),\
				np.amin(appr_wgafr, axis=0, keepdims=True)), axis=0)
stat_wcga = np.concatenate((\
				np.amax(appr_wcga, axis=0, keepdims=True),\
				np.mean(appr_wcga, axis=0, keepdims=True),\
				np.amin(appr_wcga, axis=0, keepdims=True)), axis=0)
stat_reg = np.concatenate((\
				np.amax(appr_reg, axis=0, keepdims=True),\
				np.mean(appr_reg, axis=0, keepdims=True),\
				np.amin(appr_reg, axis=0, keepdims=True)), axis=0)

# save variables
save_vars(example, appr_rwrga, itr_rwrga, appr_wgafr, itr_wgafr, appr_wcga, itr_wcga, appr_reg)
# plot approximations
plot_ga(example, stat_rwrga, stat_wgafr, stat_wcga, stat_reg)


