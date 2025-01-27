
import tensorflow as tf
import numpy as np
import logging


logger = logging.getLogger('__main__')


def cat_sample(prob_nk):
    """
    Sample from the probability distribution.
    """
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def line_search(f, x, max_step):
    """
    Simple line search algorithm. That is, having objective f and initial value
    x search along the max_step vector (shrinking it exponentially) until we find
    an improvement in f. Start with a max step and shrink it exponentially until
    there is an improvement.
    """
    max_shrinks = 100
    shrink_multiplier = 0.9
    fval = f(x)
    step_frac = 1.0
    while max_shrinks > 0:
        xnew = x + step_frac * max_step
        newfval = f(xnew)
        if fval - newfval > 0:
          return xnew
        else:
          max_shrinks -= 1
          step_frac *= shrink_multiplier
    logger.info("Can not find an improvement with line search")
    return x

def line_search_expected_improvement(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Implements a conjugate gradient algorithm. In short, solves
    Ax = b for x having only a function x -> Ax (f_Ax) and b.
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def discount_rewards(r, gamma):
    """
    Take 1D float array of rewards and compute discounted reward.
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r