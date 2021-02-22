
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy.stats import ortho_group
from methods import FrankWolfe
from methods import ContrNewton
from oracles import create_log_sum_exp_oracle


def RunExperiment(n, m, mu, fw_iters, cn_iters, cn_inner_iters, 
                  save=True, c=1.0):
    """
    Run Frank-Wolfe and Contracting Newton methods for a particular
    ('n', 'm', 'mu')-instance of random problem. Plot and save the graphs.
    """

    np.random.seed(31415)
    A = np.random.rand(n, m) * 2 - 1
    b = np.random.rand(m) * 2 - 1

    oracle = create_log_sum_exp_oracle(A.T, b, mu)
    
    x_0 = np.zeros(n)
    mu_str = str(mu)
    c_str = str(c)

    title = 'n = %d, m = %d, mu = %s (c = %s)' % (n, m, mu_str, c_str)
    filename = 'plots/%d_%d_%s_c%s' % (n, m, mu_str.split('.')[-1], 
                                        '_'.join(c_str.split('.')))
    
    print('Experiment: %s' % title)
    print('Filename: %s' % filename)
    
    start_timestamp = datetime.now()
    print('FW ...', end=' ', flush=True)
    fw = FrankWolfe(oracle, x_0, n_iters=fw_iters)
    print('DONE. Time: %s' % (str(datetime.now() - start_timestamp)))
    
    start_timestamp = datetime.now()
    print('CN ...', end=' ', flush=True)
    cn = ContrNewton(oracle, x_0, n_iters=cn_iters, inner_iters=cn_inner_iters, 
                     c=c)
    print('DONE. Time: %s' % (str(datetime.now() - start_timestamp)))
    
    fw_skip = fw_iters // 100
    
    plt.figure(figsize=(5, 4))
    fw_func = np.array(fw['func'])
    cn_func = np.array(cn['func'])
    mn2 = min(np.min(fw_func), np.min(cn_func))
    fw_res = fw_func - mn2
    cn_res = cn_func - mn2
    
    plt.semilogy(list(range(0, fw_iters+1, fw_skip)), 
                 fw_res[::fw_skip], ':', label='Frank-Wolfe', linewidth=4)
    plt.semilogy(cn_res[0:-1], label='Contr.Newton', color='red', linewidth=2)
    plt.grid()
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Function value', fontsize=14)
    plt.title(title, fontsize=14)
    t = plt.legend(fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(filename + '_iters.pdf')
        
    plt.figure(figsize=(5, 4))
    plt.semilogy(fw['time'][::fw_skip], fw_res[::fw_skip], ':', 
                 label='Frank-Wolfe', linewidth=4)
    plt.semilogy(cn['time'][0:-1], cn_res[0:-1], label='Contr.Newton', 
                 color='red', linewidth=2)
    plt.grid()
    plt.ylabel('Function value', fontsize=14)
    plt.xlabel('Time, s', fontsize=14)
    plt.title(title, fontsize=14)
    t = plt.legend(fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(filename + '_time.pdf')
    
    plt.figure(figsize=(5, 4))
    plt.plot(cn['t'])


def RunExpGroup(mu = 0.1):
    """
    Run a group of experiments with a particular smoothing parameter 'mu'
    and different 'n', 'm', 'c'.
    """
    RunExperiment(n=100, m=1000, mu=mu, fw_iters=5000, 
                  cn_iters=200, cn_inner_iters=3000, c=1.0)
    RunExperiment(n=100, m=1000, mu=mu, fw_iters=5000, 
                  cn_iters=80, cn_inner_iters=3000, c=0.05)
    RunExperiment(n=100, m=2500, mu=mu, fw_iters=5000, 
                  cn_iters=200, cn_inner_iters=3000, c=1.0) 
    RunExperiment(n=100, m=2500, mu=mu, fw_iters=5000, 
                  cn_iters=100, cn_inner_iters=3000, c=0.05)

    RunExperiment(n=500, m=1000, mu=mu, fw_iters=6000, 
                  cn_iters=200, cn_inner_iters=3000, c=1.0)
    RunExperiment(n=500, m=1000, mu=mu, fw_iters=6000, 
                  cn_iters=100, cn_inner_iters=3000, c=0.05)
    RunExperiment(n=500, m=2500, mu=mu, fw_iters=6000, 
                  cn_iters=200, cn_inner_iters=3000, c=1.0)
    RunExperiment(n=500, m=2500, mu=mu, fw_iters=6000, 
                  cn_iters=100, cn_inner_iters=3000, c=0.05)


RunExpGroup(mu = 0.1)
RunExpGroup(mu = 0.05)

