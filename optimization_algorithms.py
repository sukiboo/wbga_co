'''
	this file contains various algorithms that
	obtain a sparse (with respect to a given dictionary D)
	solution to the problem of minimizing a convex target function E
'''

import numpy as np
from scipy.optimize import minimize


# Weak Chebyshev Greedy Algorithm
def wcga(E, dE, D, max_spr, max_itr=500):

	# initialize variables
	x = np.zeros(D.shape[-1])
	itr, spr, appr, ind = 0, 0, [E(0)], []

	# iteratively construct the minimizer
	print('\nWCGA(co) optimization:')
	while itr < max_itr and spr < max_spr:

		# select element of the dictionary
		ind_max = np.argmax(np.abs(dE(x,D)))
		ind.append(ind_max)

		# construct the minimizer
		c = minimize(lambda c: E(np.matmul(c, D[ind])), np.zeros((1,spr+1))).x
		max_der = dE(x, D[ind_max])[0]
		x = np.matmul(c, D[ind])
		if np.abs(dE(x,x)[0]) > 1e-03:
			print('biorthogonality fails: {:.6f} / {:.6f}'.format(dE(x,x)[0], max_der))

		# update variables
		appr.append(E(x))
		spr += 1
		itr += 1

		# report the optimization
		print('  WCGA(co) iteration: {:3d},    solution sparcity: {:3d},    function value: {:.2e}'\
			.format(itr, spr, appr[-1]))

	return x, ind, appr


# Weak Greedy Algorithm with Free Relaxation
def wgafr(E, dE, D, max_spr, max_itr=500):

	# initialize variables
	x = np.zeros(D.shape[-1])
	itr, spr, appr, ind = 0, 0, [E(0)], []

	# iteratively construct the minimizer
	print('\nWGAFR(co) optimization:')
	while itr < max_itr:

		# select element of the dictionary
		ind_max = np.argmax(np.abs(dE(x,D)))
		if ind_max in ind or spr < max_spr:

			# construct the minimizer
			w,la = minimize(lambda c: E((1-c[0])*x + c[1]*D[ind_max]), [0,0]).x
			max_der = dE(x, D[ind_max])[0]
			x = (1-w)*x + la*D[ind_max]
			if np.abs(dE(x,x)[0]) > 1e-03:
				print('biorthogonality fails: {:.6f} / {:.6f}'.format(dE(x,x)[0], max_der))

			# update variables
			if ind_max not in ind:
				spr += 1
				appr.append(E(x))
			else:
				appr[-1] = E(x)
			ind.append(ind_max)
			itr += 1

			# report the optimization
			print('  WGAFR(co) iteration: {:3d},    solution sparcity: {:3d},    function value: {:.2e}'\
				.format(itr, spr, appr[-1]))

		else:
			itr = max_itr

	return x, ind, appr


# Rescaled Weak Relaxed Greedy Algorithm
def rwrga(E, dE, D, max_spr, max_itr=500):

	# initialize variables
	x = np.zeros(D.shape[-1])
	itr, spr, appr, ind = 0, 0, [E(0)], []

	# iteratively construct the minimizer
	print('\nRWRGA(co) optimization:')
	while itr < max_itr:

		# select element of the dictionary
		ind_max = np.argmax(np.abs(dE(x,D)))
		if ind_max in ind or spr < max_spr:

			# construct the minimizer
			la = minimize(lambda c: E(x + c*D[ind_max]), 0).x
			mu = minimize(lambda c: E(c*(x + la*D[ind_max])), 1).x
			max_der = dE(x,D[ind_max])[0]
			x = mu * (x + la*D[ind_max])
			if np.abs(dE(x,x)[0]) > 1e-03:
				print('biorthogonality fails: {:.6f} / {:.6f}'.format(dE(x,x)[0], max_der))

			# update variables
			if ind_max not in ind:
				spr += 1
				appr.append(E(x))
			else:
				appr[-1] = E(x)
			ind.append(ind_max)
			itr += 1

			# report the optimization
			print('  RWRGA(co) iteration: {:3d},    solution sparcity: {:3d},    function value: {:.2e}'\
				.format(itr, spr, appr[-1]))

		else:
			itr = max_itr

	return x, ind, appr


# Optimization with l1-regularization
def regularized_optimization(E, D, max_spr, c=.1, num_c=50, c_rate=.9):

	# initialize variables
	x_min = np.zeros(D.shape[0])
	itr, spr, appr = 0, [0], [E(0)]

	# solve minimzation problem with various regularization constants
	print('\nOptimization with l1-regularization:')
	while itr < num_c:

		# find minimizer
		x_min = minimize(lambda x: E(np.matmul(x,D)) + c*np.sum(np.abs(x)), x_min, tol=1e-03).x
		# decrease regularization constant
		c *= c_rate
		# compute solution sparsity
		s = np.sum(np.abs(x_min) > np.amax(np.abs(x_min))/100)

		# update variables
		itr += 1
		if s >= max_spr:
			itr = num_c
		elif s not in spr:
			spr.append(s)
			appr.append(E(np.matmul(x_min,D)))

		# report the optimization
		print('  iteration {:2d},    solution sparsity: {:3d},    function value: {:.2e}'\
			.format(itr, s, appr[-1]))

	# solve non-regularized problem
	x_min = minimize(lambda x: E(np.matmul(x,D)) + c*np.sum(np.abs(x)), x_min, tol=1e-03).x
	spr.append(x_min.size)
	appr.append(E(np.matmul(x_min,D)))

	# linearly interpolate optimization results
	appr = np.interp(np.arange(max_spr+1), spr, appr)

	return appr

