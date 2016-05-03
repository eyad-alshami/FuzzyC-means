from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def dist(data, centers):
    return cdist(data, centers).T


def oneStep(data, u, c, m):
    u = np.fmax(u, np.finfo(np.float64).eps)

    um = u ** m

    data = data.T

    centers = um.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = dist(data, centers)
    d = np.fmax(d, np.finfo(np.float64).eps)

    u = d ** (- 2. / (m - 1))
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

    p = 1
    while p < maxiter:
        u2 = u.copy()
        centers, u, d = oneStep(data, u2, c, m)
        p += 1

        if np.linalg.norm(u - u2) < error:
            break

    return centers, u, u0, d


colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.empty(1)
ypts = np.empty(1)
labels = np.empty(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):

    mcenters, u, u0, md = cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in mcenters:
        ax.plot(pt[0], pt[1], 'rs')

    ax.axis('off')

fig1.tight_layout()
plt.draw()
plt.show()
