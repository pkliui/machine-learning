"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()



def plot_both(X: np.ndarray, mixture_km: GaussianMixture, post_km: np.ndarray, mixture_em: GaussianMixture, post_em: np.ndarray,
         title_km: str, title_em: str):
    """Plots the mixture model for 2D data both for k-means and EM algorithms"""

    percent_km = post_km / post_km.sum(axis=1).reshape(-1, 1)
    percent_em = post_em / post_em.sum(axis=1).reshape(-1, 1)
    #
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].set_title(title_km)
    axs[0].set_xlim((-8, 10))
    axs[0].set_ylim((-8, 6))
    axs[1].set_title(title_em)
    #axs[1].set_xlim((-20, 20))
    #axs[1].set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    #
    # k-means
    for i, point in enumerate(X):
        theta = 0
        for j in range(post_km.shape[1]):
            offset = percent_km[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            axs[0].add_patch(arc)
            theta += offset
    #
    for j in range(post_km.shape[1]):
        mu = mixture_km.mu[j]
        sigma = np.sqrt(mixture_km.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        axs[0].add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        axs[0].text(mu[0], mu[1], legend)
    plt.axis('equal')
    #plt.show()
    #
    # EM-algorithm
    for i, point in enumerate(X):
        theta = 0
        for j in range(post_em.shape[1]):
            offset = percent_em[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            axs[1].add_patch(arc)
            theta += offset
    #
    for j in range(post_em.shape[1]):
        mu = mixture_em.mu[j]
        sigma = np.sqrt(mixture_em.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        axs[1].add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        axs[1].text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    raise NotImplementedError
