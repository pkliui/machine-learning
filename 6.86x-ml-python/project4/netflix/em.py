"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


# def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
#     """E-step: Softly assigns each datapoint to a gaussian component
#
#     Args:
#         X: (n, d) array holding the data, with incomplete entries (set to 0)
#         mixture: the current gaussian mixture
#         mixture's elements:
#         mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
#         var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
#         prob: np.ndarray  # (K, ) array = each row corresponds to the components' probability
#
#     Returns:
#         np.ndarray: (n, K) array holding the soft counts
#             for all components for all examples
#         float: log-likelihood of the assignment
#
#     """
#     # get current Gaussian mixture components
#     mu, var, prob = mixture
#     K = prob.shape[0]
#     n = X.shape[0]
#
#     # create the array post_prob to keep posterior probabilites
#     post_prob_nom = np.zeros((X.shape[0], prob.shape[0]))      # n,K
#
#     # posterior probability for each datapoint x_i to belong to Gaussian g
#     # for each cluster
#     for ii in range(n):
#         for jj in range(K):
#             # n,K
#             post_prob_nom[ii, jj] = prob[jj] * 1/(2*np.pi*np.sqrt(var[jj])) * np.exp(- ((X[ii,:] - mu[jj,:])**2).sum() / var[jj])
#
#     post_prob_denom = np.sum(post_prob_nom, axis=1)
#     log_likelyhood = np.sum(np.log(np.sum(post_prob_denom, axis = 1)), axis = 0)
#
#     post_prob = post_prob_nom / post_prob_denom
#
#
#     return post_prob, log_likelyhood

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

        """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        for j in range(K):
            likelihood = gaussian(X[i], mixture.mu[j], mixture.var[j])
            post[i, j] = mixture.p[j] * likelihood
        total = post[i, :].sum()
        post[i, :] = post[i, :] / total
        ll += np.log(total)

    return post, ll


def gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return np.exp(log_prob)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d = X.shape

    _, K = post.shape # n,K

    # sum up probabilities of the points that belong to cluster k
    # (1,K)
    n_hat = post.sum(axis = 0)

    # compute new probability ("weight") for cluster component k by normalizing the n_hat
    new_prob = n_hat / n

    # compute new mean and variance
    mu = np.zeros((K,d))
    var = np.zeros(K)

    for jj in range(K):
        # Computing mean
        mu[jj, :] = (X * post[:, jj, None]).sum(axis=0) / n_hat[jj]
        # Computing variance
        sse = ((mu[jj] - X) ** 2).sum(axis=1) @ post[:, jj]
        var[jj] = sse / (d * n_hat[jj])

    return GaussianMixture(mu,var,new_prob)





def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    ll = None
    while (prev_ll is None or ll - prev_ll > 1e-6 * np.abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
