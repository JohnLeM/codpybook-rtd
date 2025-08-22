"""
=========================================================
6.2 Supervised learning: benchmarks of methods with MNIST
=========================================================

Here we reproduce the results of chapter 6.2.2 - Classification problem: handwritten digits.
We will compare different models with codpy for a classification task, scoring models on unseen data.
"""

#########################################################################
# Necessary Imports
# ------------------------
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from codpydll import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import codpy.core as core
from codpy.kernel import KernelClassifier
from codpy.plot_utils import multi_plot

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
# Classification Models
# ------------------------
# Below we define the 3 classification models which will be used in our experiments.
# We use the accuracy loss as a metric along with Maximum Mean Discrepancy (MMD) for the codpy model to showcase the inverse relationship between the score and MMD.
# For classification tasks, use KernelClassifier.
# This is used the same way as Kernel, but it returns a softmax distribution over classes.
def codpy_model(x, fx, z, fz):
    kernel = KernelClassifier(
        x=x,
        fx=fx,
        clip=None,  # Clipping is used for non-probability inputs
    )
    preds = kernel(z)
    end_time = time.perf_counter()
    # mmd = np.sqrt(kernel.discrepancy(z))
    mmd = 0

    return preds, fz, mmd, end_time


def codpy_convolutional_model(x, fx, z, fz):
    class myclassifier(KernelClassifier):
        def get_kernel(self) -> callable:
            if not hasattr(self, "kernel"):
                cd.set_kernel("tinv_poly_kernel", {"p": str(1)})
                # ones = "1;" * (x.shape[0] - 1) + "1"
                # cd.set_kernel("tinv_maternnorm")
                cd.kernel_interface.set_polynomial_order(0)
                cd.kernel_interface.set_regularization(1e-9)
                cd.kernel_interface.set_map("scale_to_unitcube")
                self.kernel = core.KerInterface.get_kernel_ptr()
            return self.kernel

    kernel = myclassifier(
        x=x,
        fx=fx,
        clip=None,  # Clipping is used for non-probability inputs
    )
    preds = kernel(z)
    end_time = time.perf_counter()

    # mmd = np.sqrt(kernel.discrepancy(z))
    mmd = 0

    return preds, fz, mmd, end_time


def random_forest_model(x, fx, z, fz):
    # Convert inputs to numpy arrays and one-hot to labels
    x = np.array(x)
    fx_labels = np.argmax(np.array(fx), axis=1)
    z = np.array(z)
    fz_labels = np.argmax(np.array(fz), axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, fx_labels)
    results = model.predict(z)

    num_classes = 10  # MNIST has 10 classes (0-9)
    results_oh = np.zeros((len(results), num_classes))
    results_oh[np.arange(len(results)), results] = 1

    # Also convert true labels to one-hot for consistent return
    fz_oh = np.zeros((len(fz_labels), num_classes))
    fz_oh[np.arange(len(fz_labels)), fz_labels] = 1

    return results_oh, fz_oh, None, time.perf_counter()


def torch_model(x, fx, z, fz):
    x = torch.tensor(x, dtype=torch.float32)
    fx = torch.tensor(fx, dtype=torch.long).argmax(dim=1)
    z = torch.tensor(z, dtype=torch.float32)
    fz = torch.tensor(fz, dtype=torch.long)
    out_shape = fz.shape[1]
    # batch_size = max(x.shape[0] // 4, 32)

    dataset = TensorDataset(x, fx)
    trainloader = DataLoader(dataset, batch_size=min(32, x.shape[0]), shuffle=True)

    class FFN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    model = FFN(x.shape[1], out_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        for x_batch, fx_batch in trainloader:
            pred = model(x_batch)
            loss = loss_fn(pred, fx_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        results = model(z)

    return results.cpu(), fz.cpu(), None, time.perf_counter()


def torch_model_naive(x, fx, z, fz):
    x = torch.tensor(x, dtype=torch.float32)
    fx = torch.tensor(fx, dtype=torch.float32).argmax(dim=1)
    z = torch.tensor(z, dtype=torch.float32)
    fz = torch.tensor(fz, dtype=torch.float32)
    out_shape = fz.shape[1]

    # batch_size = x.shape[0] // 4
    dataset = TensorDataset(x, fx)
    trainloader = DataLoader(dataset, batch_size=min(32, x.shape[0]), shuffle=True)

    class FFN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    model = FFN(x.shape[1], out_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(128):
        for x_batch, fx_batch in trainloader:
            pred = model(x_batch)
            loss = loss_fn(pred, fx_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        results = model(z)
    return results, fz, None, time.perf_counter()


def vgg_torch_model(x, fx, z, fz):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1, 28, 28)
    fx = torch.tensor(fx, dtype=torch.float32).argmax(dim=1).long()

    z = torch.tensor(z, dtype=torch.float32).view(-1, 1, 28, 28)
    fz = torch.tensor(fz, dtype=torch.float32).long()  # .argmax(dim=1).long()
    out_shape = fz.shape[1]

    dataset = TensorDataset(x, fx)
    trainloader = DataLoader(dataset, batch_size=min(128, x.shape[0]), shuffle=True)

    class VGG(nn.Module):
        def __init__(self, input_dim=1, output_dim=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_dim, out_channels=32, kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(0.3),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = VGG(x.shape[1], out_shape)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        for x_batch, fx_batch in trainloader:
            pred = model(x_batch)
            loss = loss_fn(pred, fx_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        results = model(z)
    return results, fz, None, time.perf_counter()


#########################################################################
# Gathering Results
# ------------------------
# We train the models on different subsets of the training dataset.
# We always evaluate the models on the entire test set.

#########################################################################

results = {}
times = {}
mmds = {}
models = [
    torch_model,
    torch_model_naive,
    random_forest_model,
    codpy_convolutional_model,
    codpy_model,
    vgg_torch_model,
]

X, FX, Z, FZ = get_MNIST_data()
# torch.manual_seed(0)
# idxs = torch.randperm(len(X))
# idxs_z = torch.randperm(len(Z))
SIZE = 2**11
# X, FX, Z, FZ = X[idxs][:SIZE], FX[idxs][:SIZE], Z[idxs_z][:SIZE], FZ[idxs_z][:SIZE]
length_ = len(X)
scenarios_list = [
    (int(i), int(i), -1, -1) for i in 2 ** np.arange(np.log2(16), np.log2(SIZE) + 1)
]

for scenario in scenarios_list:
    Nx, Nfx, Nz, Nfz = scenario
    x, fx, z, fz = X[:Nx], FX[:Nfx], Z[:Nz], FZ[:Nfz]
    results[len(x)] = {}
    times[len(x)] = {}
    mmds[len(x)] = {}
    for model in models:
        start_time = time.perf_counter()
        logits, target, mmd, end_time = model(x, fx, z, fz)
        mmds[len(x)][model.__name__] = mmd
        pred_classes = np.argmax(logits, axis=1)
        true_classes = np.argmax(target, axis=1)
        accuracy = accuracy_score(true_classes, pred_classes)
        results[len(x)][model.__name__] = accuracy
        times[len(x)][model.__name__] = end_time - start_time
        print(
            f"Model: {model.__name__}, size: {Nx}, Time taken: {times[len(x)][model.__name__]:.4f} seconds, Accuracy: {results[len(x)][model.__name__]:.4f} seconds"
        )

res = [{"data": results}, {"data": times}]
# res = [{"data": results}, {"data": mmds}, {"data": times}]


#########################################################################
# Plotting
# ------------------------
#########################################################################
def plot_one(inputs):
    results = inputs["data"]
    ax = inputs["ax"]
    legend = inputs["legend"]

    model_aliases = {
        "torch_model": "FFN",
        "torch_model_naive": "FFN: basic",
        "codpy_model": "KRR",
        "codpy_convolutional_model": "CKRR",
        "random_forest_model": "RF",
        "vgg_torch_model": "VGG",
    }

    model_markers = {
        "torch_model": "o",
        "torch_model_naive": "s",
        "codpy_model": "D",
        "codpy_convolutional_model": "^",
        "random_forest_model": "v",
        "vgg_torch_model": "X",
    }

    x_vals = sorted(results.keys())
    for model_name in next(iter(results.values())).keys():
        y_vals = [results[x][model_name] for x in x_vals]
        label = model_aliases.get(model_name, model_name)
        marker = model_markers.get(model_name, "o")
        ax.plot(x_vals, y_vals, marker=marker, label=label)

    ax.set_xlabel("Number of Training Examples", fontsize=14)
    ax.set_ylabel(legend, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    return ax


multi_plot(
    res,
    plot_one,
    mp_nrows=1,
    mp_ncols=2,  # 3
    legends=["Accuracy", "Times"],
    # legends=["Accuracy", "MMD", "Times"],
    mp_figsize=(14, 10),
)
plt.show()
pass
