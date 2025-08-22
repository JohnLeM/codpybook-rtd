"""
=============================================
6.3 Unsupervised learning: Clustering - MNIST
=============================================

We show how to reproduce the results of the chapter 6.3.2 - Application to supervised machine learning - Classification problem: handwritten digits of the book.
We will compare the codpy MMD minimization-based algorithm with scikit learn k-means in an unsupervised setting.
The goal is to show the different scores as we increase the number of centroids Ny used for clustering.
"""

#########################################################################
# Necessary Imports
# ------------------------
#########################################################################
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "4"

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codpy.clustering import GreedySearch, SharpDiscrepancy

# We use a custom hot encoder for performances reasons.
from codpy.kernel import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, pairwise_distances
from torchvision import datasets, transforms

try:
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "data")
except NameError:
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)

#########################################################################
# MNIST Data Preparation
# ------------------------
# We normalize pixel values and reshape data for processing.
# Pixel data is used as flat vectors, and labels are one-hot encoded.


#########################################################################
def get_MNIST_data(N=-1, flatten=True, one_hot=True, seed=43):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root=".", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=".", train=False, download=True, transform=transform
    )

    x = train_data.data.numpy().astype(np.float32) / 255.0
    fx = train_data.targets.numpy()

    z = test_data.data.numpy().astype(np.float32) / 255.0
    fz = test_data.targets.numpy()

    if flatten:
        x = x.reshape(len(x), -1)
        z = z.reshape(len(z), -1)

    if one_hot:
        fx = np.eye(10)[fx]
        fz = np.eye(10)[fz]
    if N == -1:
        N = x.shape[0]
    indices = random.sample(range(x.shape[0]), min(N, x.shape[0]))
    x, fx = x[indices], fx[indices]

    return x, fx, z, fz


#########################################################################
# Clustering Models
# ------------------------
# This section defines the K-means and CodPy clustering models.
# We wrap the clustering methods with kernels assigning labels to clusters based on the target values `fx`.
# This allow us to compute score metrics for the models.


#########################################################################
def codpy_sharp(x, fx, Ny):
    # Select the codpy model for clustering and select centers
    greedy_search = SharpDiscrepancy(x=x, N=Ny)
    centers = greedy_search.cluster_centers_
    # Set a classifier which will assign labels to clusters based on fx
    kernel = KernelClassifier(x=x, y=centers, fx=fx)
    kernel.set_kernel_ptr(greedy_search.k.kernel)
    return centers, kernel


def codpy_clustering(x, fx, Ny):
    # Select the codpy model for clustering and select centers
    greedy_search = GreedySearch(x=x, N=Ny)
    centers = greedy_search.cluster_centers_
    # Set a classifier which will assign labels to clusters based on fx
    kernel = KernelClassifier(x=x, y=centers, fx=fx)
    kernel.set_kernel_ptr(greedy_search.k.kernel)
    return centers, kernel


def kmeans_clustering(x, fx, Ny):
    # Use Kmeans from sklearn to get the clusters
    if Ny >= x.shape[0]:
        centers = x
    else:
        centers = KMeans(n_clusters=Ny, random_state=1).fit(x).cluster_centers_
    # Set a classifier which will assign labels to centers based on fx
    kernel = KernelClassifier(x=x, y=centers, fx=fx)
    return centers, kernel

def compute_mmd(x_test, z):
    kernel = Kernel(x=x_test, order=0)
    mmd = kernel.discrepancy(z)
    return mmd


def compute_inertia(x, y):
    return np.sum((pairwise_distances(x, y) ** 2).min(axis=1))


