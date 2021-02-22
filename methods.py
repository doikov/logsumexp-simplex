import numpy as np

from collections import defaultdict
from datetime import datetime

from oracles import OracleCallsCounter


def FrankWolfe(oracle, x_0, n_iters=1000, trace=True):
    """
    Run Frank-Wolfe algorithm for 'n_iters' iterations, minimizing
    smooth function 'f' over the standard Simplex.
    
    'oracle' is an instance of BaseSmoothOracle for the objective 'f'.
    """
    
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()
    
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    
    history = defaultdict(list)
    
    for k in range(n_iters + 1):
        
        if trace:
            history['func'].append(f_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if k == n_iters:
            break

        i = np.argmin(g_k)
        
        # Stepsize policy.
        gamma_k = 2.0 / (k + 2.0)

        x_k *= (1 - gamma_k)
        x_k[i] += gamma_k
        
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        
    return history


def ContrNewton(oracle, x_0, n_iters=1000, 
                inner_iters=1000, c=1.0, trace=True):
    """
    Run Inexact Contracting Newton method for 'n_iters' iterations, minimizing
    smooth function 'f' over the standard Simplex.

    At each iteration, we use Conditional Gradient method (a modification 
    of Frank-Wolfe) to solve the inner subproblem inexactly. The required 
    accuracy level is bounded by the Estimating Function (duality gap).
    'c' is a flexible parameter controlling the inner accuracy.

    'oracle' is an instance of BaseSmoothOracle for the objective 'f'.
    """
    
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()
    
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    Hess_k = oracle.hess(x_k)
    
    history = defaultdict(list)
    
    for k in range(n_iters + 1):

        if trace:
            history['func'].append(f_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
        
        if k == n_iters:
            break

        # Stepsize policy.
        gamma_k = 3.0 / (k + 3.0)
        
        # Initialization of the inner method.
        
        # Components of the Estimating Function.
        phi_t_const = 0.0
        phi_t_grad = np.zeros_like(x_0)

        # Starting point z_0 := x_k.
        z_t = np.copy(x_k)
        
        # G(z) = <g_k, z - x_k> + 0.5 * gamma_k * <Hess(z - x_k), z - x_k>.
        # G'(z) = g_k + gamma_k * Hess(z - x_k).
        Hess_k_x_k = Hess_k.dot(x_k)

        # h_t := z_t - x_k.
        h_t = np.zeros_like(x_k)
        G_prime_z = np.copy(g_k)
        G_z = 0.5 * (g_k + G_prime_z).dot(h_t)
        
        for t in range(inner_iters):

            # Stepsize policy for the inner method.
            alpha_t = 2.0 / (t + 2.0)

            phi_t_const *= (1 - alpha_t)
            phi_t_const += alpha_t * (G_z - G_prime_z.dot(z_t))
            phi_t_grad *= (1 - alpha_t)
            phi_t_grad += alpha_t * G_prime_z
            
            i = np.argmin(phi_t_grad)
            # w_{t + 1} = (0, 0, 0, ..., 0, 1, 0, ..., 0)
            #                           >>> i <<<
            
            z_t *= (1 - alpha_t)
            z_t[i] += alpha_t
            
            phi_w = phi_t_const + phi_t_grad[i]
            
            h_t = z_t - x_k

            G_prime_z *= (1 - alpha_t)
            G_prime_z += alpha_t * (g_k + gamma_k * (Hess_k[:, i] - Hess_k_x_k))
        
            G_z = 0.5 * (g_k + G_prime_z).dot(h_t)

            # Check for the stopping condition for the inner method.
            if G_z - phi_w <= c * gamma_k ** 2 + 1e-8:
                break
            
        x_k *= (1 - gamma_k)
        x_k += gamma_k * z_t
        
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        Hess_k = oracle.hess(x_k)
        
        if trace:
            history['t'].append(t)
    
    return history

