
import math
import numpy as np
import scipy
from scipy.special import logsumexp, softmax


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of the function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def third_vec_vec(self, x, v):
        """
        Computes tensor-vector-vector product with the third derivative tensor
        D^3 f(x)[v, v].
        """
        raise NotImplementedError('Third derivative oracle is not implemented.')


class OracleCallsCounter(BaseSmoothOracle):
    """
    Wrapper to count oracle calls.
    """
    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)
        
    def hess(self, x):
        self.hess_calls += 1
        return self.oracle.hess(x)

    def hess_vec(self, x, v):
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x, v):
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)


class LogSumExpOracle(BaseSmoothOracle):
    """
    Oracle for function:
        func(x) = mu log sum_{i=1}^m exp( (<a_i, x> - b_i) / mu )
        a_1, ..., a_m are rows of (m x n) matrix A.
        b is given (m x 1) vector.
        mu is a scalar value.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, mu):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.copy(b)
        self.mu = mu
        self.mu_inv = 1.0 / mu
        self.last_x = None
        self.last_x_pi = None
        self.last_v = None

    def func(self, x):
        self._update_a(x)
        return self.mu * logsumexp(self.a)

    def grad(self, x):
        self._update_a_and_pi(x)
        return self.AT_pi

    def hess(self, x):
        self._update_a_and_pi(x)
        return self.mu_inv * (self.matmat_ATsA(self.pi) - \
            np.outer(self.AT_pi, self.AT_pi.T))

    def hess_vec(self, x, v):
        self._update_hess_vec(x, v)
        return self._hess_vec

    def third_vec_vec(self, x, v):
        self._update_hess_vec(x, v)
        return self.mu_inv * (
            self.mu_inv * self.matvec_ATx(
                self.pi * (self.Av * (self.Av - self.AT_pi_v))) -
            self.AT_pi_v * self._hess_vec -
            self._hess_vec.dot(v) * self.AT_pi)

    def _update_a(self, x):
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.mu_inv * (self.matvec_Ax(x) - self.b)

    def _update_a_and_pi(self, x):
        self._update_a(x)
        if not np.array_equal(self.last_x_pi, x):
            self.last_x_pi = np.copy(x)
            self.pi = softmax(self.a)
            self.AT_pi = self.matvec_ATx(self.pi)

    def _update_hess_vec(self, x, v):
        if not np.array_equal(self.last_x, x) or \
           not np.array_equal(self.last_v, v):
            self._update_a_and_pi(x)
            self.last_v = np.copy(v)
            self.Av = self.matvec_Ax(v)
            self.AT_pi_v = self.AT_pi.dot(v)
            self._hess_vec = self.mu_inv * ( \
                self.matvec_ATx(self.pi * self.Av) - \
                self.AT_pi_v * self.AT_pi)


def create_log_sum_exp_oracle(A, b, mu):
    """
    Auxiliary function for creating log-sum-exp oracle.
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    B = None

    def matmat_ATsA(s):
        nonlocal B
        if B is None: B = A.toarray() if scipy.sparse.issparse(A) else A
        return B.T.dot(B * s.reshape(-1, 1))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)

