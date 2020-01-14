'''
	this file contains various auxiliary functions
'''

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pickle
import time


# define norming functional in l_p-space
def F_lp(x, y, p):
	# compute and normalize norming functional
	F = np.matmul(y.reshape((-1,x.size)), (np.sign(x) * np.abs(x)**(p-1)).reshape((-1,1)))
	F /= 1e-07 + norm(x,p)**(p-1)
	return F.ravel()


# construct random dictionary
def random_dictionary(num_dict, dim):
	# generate dictionary
	D = np.random.rand(num_dict, dim)
	# normalize dictionary in 1-norm
	D /= norm(D, 1, axis=1, keepdims=True)
	return D


# construct canonical dictionary
def canonical_dictionary(dim):
	# generate dictionary
	D = np.identity(dim)
	return D


# construct random sparse element with respect to the dictionary
def random_element(D, f_spr):
	# select support from the dictionary
	f_supp = D[np.random.randint(0, D.shape[0], f_spr)]
	# generate random coefficients
	f_coeff = np.random.randn(1, f_spr)
	# compute the element
	f = np.matmul(f_coeff, f_supp)
	return f.ravel()


# save variables
def save_vars(example, *data):
	# construct filename and save variables
	filename = time.strftime('%H.%M.%S', time.localtime()) + ' ' + example
	pickle.dump(data, open('./saves/{:s}.save'.format(filename), 'wb'))
	return


# plot greedy algorithms
def plot_ga(example, stat_rwrga, stat_wgafr, stat_wcga, stat_reg):
	# set up figure
	plt.figure(figsize=(10,5))
	plt.xlim(0, stat_reg.shape[1]-1)
	plt.xlabel('solution sparsity')
	plt.ylabel('function value')
	plt.yscale('log')
	# define colors
	color_fill = ['dodgerblue', 'deeppink', 'turquoise', 'gold']
	color_line = ['blue', 'red', 'forestgreen', 'darkorange']
	# plot statistical distribution
	x_spr = np.arange(stat_reg.shape[1])
	plt.fill_between(x_spr, stat_rwrga[0], stat_rwrga[2], alpha=.25, color=color_fill[0])
	plt.fill_between(x_spr, stat_wgafr[0], stat_wgafr[2], alpha=.25, color=color_fill[1])
	plt.fill_between(x_spr, stat_wcga[0], stat_wcga[2], alpha=.25, color=color_fill[2])
	if 0 not in stat_reg:
		plt.fill_between(x_spr, stat_reg[0], stat_reg[2], alpha=.25, color=color_fill[3])
	# plot average values
	plt.plot(x_spr, stat_rwrga[1], linewidth=3, color=color_line[0], label='rwrga')
	plt.plot(x_spr, stat_wgafr[1], linewidth=3, color=color_line[1], label='wgafr')
	plt.plot(x_spr, stat_wcga[1], linewidth=3, color=color_line[2], label='wcga')
	if 0 not in stat_reg:
		plt.plot(x_spr, stat_reg[1], linewidth=3, color=color_line[3], label='l1-reg')
	# show the figure
	plt.legend(loc='lower left')
	plt.tight_layout()
	filename = time.strftime('%H.%M.%S', time.localtime()) + ' ' + example
	plt.savefig('./saves/{:s}.pdf'.format(filename), dpi=300, format='pdf')
	plt.show()
	return

