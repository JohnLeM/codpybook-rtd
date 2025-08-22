"""
====================================================
6.6 Large scale datasets
====================================================


We show how to reproduce the results of the chapter 6.3.5 - Application to supervised machine learning - Large scale dataset of the book.
We illustrate the behavior of multiple clustering methods on the MNIST dataset to perform large scale supervised learning.
"""

#########################################################################
# Necessary Imports
# ------------------------
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from codpy.clustering import (
    BalancedClustering,
    GreedySearch,
    MiniBatchkmeans,
    RandomClusters,
    SharpDiscrepancy,
)
from codpy.kernel import KernelClassifier
from codpy.plot_utils import multi_plot
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

#########################################################################
# MNIST Data Preparation
# ------------------------
# We normalize pixel values and reshape data for processing.
# Pixel data is used as flat vectors, and labels are one-hot encoded.


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
# This section defines the clutering models used to perform large scale supervised learning.
# This is done as a "divide and conquer" strategy,
# where the dataset is split into smaller groups (clusters).
# We present here different clustering methods that can be used with CodPy.
# Each method is wrapped into `BalancedClustering`, which allows to manage clusters as
# we add new data.

def greedy(x, fx, Ny):
    n_batch = x.shape[0] // Ny
    clust = lambda x, N, **kwargs: BalancedClustering(GreedySearch, x=x, N=N)
    kernel = KernelClassifier(x=x, fx=fx, n_batch=n_batch, set_clustering=clust)
    return kernel


def k_means(x, fx, Ny):
    n_batch = x.shape[0] // Ny
    clust = lambda x, N, **kwargs: BalancedClustering(MiniBatchkmeans, x=x, N=N)
    kernel = KernelClassifier(x=x, fx=fx, n_batch=n_batch, set_clustering=clust)
    return kernel


def sharpDisc(x, fx, Ny):
    n_batch = x.shape[0] // Ny
    clust = lambda x, N, **kwargs: BalancedClustering(SharpDiscrepancy, x=x, N=N)
    kernel = KernelClassifier(x=x, fx=fx, n_batch=n_batch, set_clustering=clust)
    return kernel


def randomClusters(x, fx, Ny):
    n_batch = x.shape[0] // Ny
    clust = lambda x, N, **kwargs: BalancedClustering(RandomClusters, x=x, N=N)
    kernel = KernelClassifier(x=x, fx=fx, n_batch=n_batch, set_clustering=clust)
    return kernel


#########################################################################
# Running the Experiment
# ------------------------
# This section runs the experiments comparing accuracy scores for different clustering strategies.


results = {}
times = {}
models = [greedy, k_means, sharpDisc, randomClusters]

SIZE = 2**8
X, FX, Z, FZ = get_MNIST_data(SIZE)
length_ = len(X)
Nys = [5, 10]


def get_score(z, fz, predictor):
    # The score is accuracy
    f_z = predictor(z)
    f_z = f_z.argmax(1)
    ground_truth = fz.argmax(axis=-1)
    out = confusion_matrix(ground_truth, f_z)
    score = np.trace(out) / np.sum(out)
    return score


for Ny in Nys:
    results[Ny] = {}
    times[Ny] = {}
    for model in models:
        start_time = time.perf_counter()
        kernel = model(X, FX, Ny)
        score = get_score(Z, FZ, kernel)
        results[Ny][model.__name__] = score
        end_time = time.perf_counter()
        times[Ny][model.__name__] = end_time - start_time
        print(
            f"Model: {model.__name__}, Time taken: {times[Ny][model.__name__]:.4f} seconds"
        )

res = [{"data": results}, {"data": times}]

#########################################################################
# Plotting
# ------------------------
# This section formats data plots the different experiments on a figure.



def plot_one(inputs):
    results = inputs["data"]
    ax = inputs["ax"]
    legend = inputs["legend"]
    for model_name in next(iter(results.values())).keys():
        x_vals = sorted(results.keys())
        y_vals = [results[x][model_name] for x in x_vals]
        ax.plot(x_vals, y_vals, marker="o", label=model_name)
    ax.set_xlabel("Ny")
    ax.set_ylabel(legend)
    ax.legend()
    ax.grid(True)
    return ax


multi_plot(
    res,
    plot_one,
    mp_nrows=1,
    mp_ncols=2,
    legends=["Scores", "Times"],
    mp_figsize=(14, 10),
)
plt.show()
