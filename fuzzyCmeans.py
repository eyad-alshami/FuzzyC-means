import numpy as np
from scipy.spatial.distance import cdist


def dist(data, centers):
    return cdist(data, centers).T

def oneStep(data, u, c, m):
    u = np.fmax(u, np.finfo(np.float64).eps)

    um = u** m

    data = data.T

    centers = um.dot(data) / (np.ones((data.shap[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = dist(data, centers)
    d = np.fmax(d, np.finfo(np.float64).eps)

    u = d** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return centers, u, d


def cmeans(data, c, m, maxiter, error, init=None, seed=None):
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones((c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    p= 1
    while p < maxiter:
        u2 = u.copy()
        centers, u, d = oneStep(data, u2, c, m)
        p += 1

        if np.linalg.norm(u - u2) < error:
            break

    return centers, u, u0, d








