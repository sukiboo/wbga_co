'''
    This file contains the code for the numerical experiments presented in the paper
    'Biorthogonal greedy algorithms in convex optimization', see https://arxiv.org/abs/2001.05530
    The examples presented in the paper can be recreated by running the code
    with an appropriate parameter, i.e. ex = 1,2,3,4 respectively
'''

from setup_experiments import SetupExperiments
import argparse


def parse_arguments():
    '''parse the arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--experiment', default=1, help='which numerical example to run: 1, 2, 3, or 4')
    parser.add_argument('-s', '--seed', default=0, help='value of random seed, default is 0')
    parser.add_argument('-n', '--num_exp', default=100, help='number of simulations to run, default is 100')
    args = parser.parse_args()
    return int(args.experiment), int(args.seed), int(args.num_exp)


if __name__ == '__main__':
    '''setupe and run experiments'''
    ex, seed, num_exp = parse_arguments()

    if ex in [1,2,3,4]:
        print(f'Performing {num_exp} simulations of numerical example {ex}...')
        exp = SetupExperiments(ex=ex, seed=seed)
        exp.run(num_exp=num_exp)

        # visualize the result
        exp.plot_error_sparsity()
        exp.plot_error_iterations()

    else:
        raise SystemExit(f'\nexample {ex} is not implemented, available examples: 1,2,3,4\n')

