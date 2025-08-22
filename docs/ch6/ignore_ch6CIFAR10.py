"""
==========================================================================================
6.2 Supervised learning: Efficiency analysis of algorithms with CIFAR-10
==========================================================================================

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
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import codpy.core
from codpy.data_processing import hot_encoder
from codpy.kernel import KernelClassifier
from codpy.plot_utils import multi_plot

#########################################################################
# MNIST Data Preparation
# ------------------------
# We normalize pixel values and reshape data for processing.
# Pixel data is used as flat vectors, and labels are one-hot encoded.


def get_CIFAR10_data(N=-1):
    from torchvision import transforms as transforms

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    x, fx = train_set.data, np.array(train_set.targets)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    z, fz = test_set.data, np.array(test_set.targets)

    if N != -1:
        indices = random.sample(range(x.shape[0]), N)
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
    x, z = x / 255.0, z / 255.0
    x, z, fx, fz = (
        x.reshape(len(x), -1),
        z.reshape(len(z), -1),
        fx.reshape(len(fx), -1),
        fz.reshape(len(fz), -1),
    )
    fx, fz = (
        hot_encoder(
            pd.DataFrame(data=fx), cat_cols_include=[0], sort_columns=True
        ).values,
        hot_encoder(
            pd.DataFrame(data=fz), cat_cols_include=[0], sort_columns=True
        ).values,
    )

    class my_kernel(KernelClassifier):
        def __init__(
            self,
            x=x,
            fx=fx,
            clip=None,  # Clipping is used for non-probability inputs
            **kwargs,
        ):
            super().__init__(x=x, fx=fx, clip=clip, **kwargs)

        def __call__(self, z, **kwargs):
            z = codpy.core.get_matrix(z)
            if self.x is None:
                return None
                # return softmax(np.full((z.shape[0],self.actions_dim),np.log(.5)),axis=1)
            knm = super().__call__(z, **kwargs)
            return softmax(knm, axis=1)

    kernel = my_kernel(
        x=x,
        fx=fx,
        clip=None,  # Clipping is used for non-probability inputs
    )
    start_time = time.perf_counter()
    kernel.get_knm_inv()
    l_time = time.perf_counter() 
    preds = kernel(z)
    return codpy.core.get_matrix(preds).argmax(1), codpy.core.get_matrix(fz).argmax(1), l_time - start_time, time.perf_counter()-l_time


def random_forest_model(x, fx, z, fz):
    # Convert inputs to numpy arrays and one-hot to labels
    x, z, fx, fz = (
        x.reshape(len(x), -1),
        z.reshape(len(z), -1),
        fx.reshape(len(fx), -1),
        fz.reshape(len(fz), -1),
    )
    x = np.array(x)
    z = np.array(z)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        max_features="sqrt",
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    start_time = time.perf_counter()
    model.fit(x, fx.ravel())
    l_time = time.perf_counter() 
    results = model.predict(z)

    return results, fz.squeeze(), l_time - start_time, time.perf_counter()-l_time


def torch_perceptron(x, fx, z, fz):
    x, z = x / 255.0, z / 255.0
    x, z, fx, fz = (
        x.reshape(len(x), -1),
        z.reshape(len(z), -1),
        fx.reshape(len(fx), -1),
        fz.reshape(len(fz), -1),
    )
    fx, fz = (
        hot_encoder(
            pd.DataFrame(data=fx), cat_cols_include=[0], sort_columns=True
        ).values,
        hot_encoder(
            pd.DataFrame(data=fz), cat_cols_include=[0], sort_columns=True
        ).values,
    )
    out_shape = fz.shape[1]
    x = torch.tensor(x, dtype=torch.float32)
    fx = torch.tensor(fx, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)
    fz = torch.tensor(fz, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(x.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, out_shape),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    n_samples = x.shape[0]
    batch_size = n_samples // 4
    start_time = time.perf_counter()
    for epoch in range(128):
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            x_batch = x[batch_idx]
            fx_batch = fx[batch_idx]

            pred = model(x_batch)
            loss = loss_fn(pred, fx_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    l_time = time.perf_counter() 
    with torch.no_grad():
        results = model(z)
    return results.argmax(1), fz.argmax(1), l_time - start_time, time.perf_counter()-l_time


class GeneralCNN(nn.Module):
    def __init__(self, cfg, in_channels=3, num_classes=10):
        super().__init__()
        self.cfg = cfg
        self.features = self._make_feature_layers(cfg["features"], in_channels)
        flat_dim = self._infer_flatten_size(in_channels)
        self.classifier = self._make_classifier_layers(
            cfg["classifier"], flat_dim, num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_feature_layers(self, cfg_list, in_channels):
        layers = []
        for item in cfg_list:
            if isinstance(item, str):
                if item.startswith("C"):
                    out = int(item[1:])
                    layers.append(nn.Conv2d(in_channels, out, kernel_size=3, padding=1))
                    if self.cfg.get("batchnorm", False):
                        layers.append(nn.BatchNorm2d(out))
                    layers.append(nn.ReLU())
                    in_channels = out
                elif item == "R":
                    layers.append(nn.ReLU())
                elif item.startswith("M"):
                    k = int(item[1:])
                    layers.append(nn.MaxPool2d(kernel_size=k, stride=2))
                elif item == "GAP":
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                elif item.startswith("A"):
                    size = int(item[1:])
                    layers.append(nn.AdaptiveAvgPool2d((size, size)))
                else:
                    raise ValueError(f"Unknown feature layer: {item}")
            elif isinstance(item, dict) and item.get("resblock"):
                block = item["resblock"]
                layers.append(
                    ResNetBlock2(
                        in_channels, block["out"], stride=block.get("stride", 1)
                    )
                )
                in_channels = block["out"]
        return nn.Sequential(*layers)

    def _make_classifier_layers(self, cfg_list, in_features, num_classes):
        layers = []
        for item in cfg_list:
            if item == "F":
                layers.append(nn.Flatten())
            elif item == "R":
                layers.append(nn.ReLU(inplace=True))
            elif item.startswith("L"):
                in_f, out_f = item[1:].split("-")
                if in_f == "auto":
                    in_f = in_features
                else:
                    in_f = int(in_f)
                out_f = int(out_f)
                layers.append(nn.Linear(in_f, out_f))
                in_features = out_f
            elif item.startswith("D"):
                p = float(item[1:])
                layers.append(nn.Dropout(p))
        if not any(isinstance(l, nn.Linear) for l in layers):
            layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers)

    def _infer_flatten_size(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            x = self.features(dummy)
            return x.view(1, -1).shape[1]


class ConvBNRelu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding="same"
    ):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetBlock2(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"
    ):
        super().__init__()
        self.conv1 = ConvBNRelu(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvBNRelu(out_channels, out_channels, kernel_size, 1, padding)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


######################################
# Model Configs
# -----------------------

model_configs = {
    "VGG-like": {
        "features": [
            "C16",
            "C16",
            "M3",
            "C32",
            "C32",
            "M3",
            "C64",
            "C64",
            "M3",
            "C128",
            "C128",
            "GAP",
        ],
        "classifier": ["F", "L128-10"],
        "batchnorm": True,
    },
    "ResNet-like": {
        "features": [
            {"resblock": {"out": 8}},
            {"resblock": {"out": 16, "stride": 2}},
            {"resblock": {"out": 32, "stride": 2}},
            {"resblock": {"out": 64, "stride": 2}},
            "GAP",
        ],
        "classifier": ["F", "L64-10"],
    },
    "AlexNet-like": {
        "features": ["C10", "R", "M2"],
        "classifier": ["F", "Lauto-100", "R", "D0.1", "L100-10"],
    },
}


def torch_nn_model(x, fx, z, fz, model_constructor, type="MLP"):
    # x, z = x / 255.0, z / 255.0
    if fx.ndim > 1:
        fx = np.argmax(fx, axis=1)
    if fz.ndim > 1:
        fz = np.argmax(fz, axis=1)

    if type == "CNN":
        x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        z = torch.tensor(z, dtype=torch.float32).permute(0, 3, 1, 2)
    else:
        x = torch.tensor(x, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)

    fx = torch.tensor(fx, dtype=torch.long)
    fz = torch.tensor(fz, dtype=torch.long)

    model = model_constructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dataset = TensorDataset(x, fx)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    start_time = time.perf_counter()

    while True:
        model.train()
        for x_batch, fx_batch in train_loader:
            logits = model(x_batch)
            loss = loss_fn(logits, fx_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        duration = time.perf_counter() - start_time
        if duration > len(x) / 64:
            break
    l_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        results = model(z)
    p_time = time.perf_counter()
    return results.argmax(1).numpy(), fz.numpy(), l_time - start_time, p_time - l_time


###############################################
# Model wrappers
# -----------------------------------


def alexnet_tiny(x, fx, z, fz):
    return torch_nn_model(
        x, fx, z, fz, lambda: GeneralCNN(model_configs["AlexNet-like"]), type="CNN"
    )


def vgg_tiny(x, fx, z, fz,**cnn_args):
    return torch_nn_model(
        x, fx, z, fz, lambda: GeneralCNN(model_configs["VGG-like"],**cnn_args), type="CNN"
    )


def resnet_tiny(x, fx, z, fz):
    return torch_nn_model(
        x, fx, z, fz, lambda: GeneralCNN(model_configs["ResNet-like"]), type="CNN"
    )


#########################################################################
# Gathering Results
# ------------------------
# We train the models on different subsets of the training dataset.
# We always evaluate the models on the entire test set.

if __name__ == "__main__":
    results = {}
    times = {}
    efficiency = {}
    models = [
        vgg_tiny,
        torch_perceptron,
        random_forest_model,
        codpy_model,
        alexnet_tiny,
        resnet_tiny,
    ]

    X, FX, Z, FZ = get_CIFAR10_data()
    SIZE = 2**16

    stopping_time = 20  # seconds
    # stopping_time =  None

    length_ = len(X)
    scenarios_list = [
        (int(i), int(i)) for i in 2 ** np.arange(np.log2(128), np.log2(SIZE) + 1)
    ]

    results = {}
    l_times = {}
    p_times = {}
    efficiency = {}

    for model in models:
        results[model.__name__] = {}
        l_times[model.__name__] = {}
        p_times[model.__name__] = {}
        efficiency[model.__name__] = {}
        for scenario in scenarios_list:
            Nx, Nfx = scenario
            if Nx < length_:
                indices = random.sample(range(X.shape[0]), Nx)
                x, fx, z, fz = X[indices], FX[indices], Z, FZ
            else:
                x, fx, z, fz = X, FX, Z, FZ
            pred_classes, target, l_time, p_time = model(x, fx, z, fz)
            accuracy = accuracy_score(fz, pred_classes)

            results[model.__name__][Nx] = accuracy
            l_times[model.__name__][Nx] = l_time
            p_times[model.__name__][Nx] = p_time
            efficiency[model.__name__][Nx] = [l_time, accuracy]

            print(
                f"Model: {model.__name__}, Size: {Nx}, Time taken: {l_time:.4f} seconds, "
                f"Accuracy: {accuracy:.4f}"
            )
            if stopping_time is not None:
                if l_time > stopping_time:
                    break


    res = [{"data": results}, {"data": l_times}, {"data": efficiency}]


#########################################################################
# Plotting
# ------------------------
    def plot_one(inputs):
        results = inputs["data"]
        ax = inputs["ax"]
        legend = inputs["legend"]

        if legend == "Efficiency":
            for model_name in results.keys():
                x_vals = sorted(results[model_name].keys())
                y_vals = np.array([results[model_name][x] for x in x_vals])
                if y_vals[0] is None:
                    continue
                ax.plot(y_vals[:, 0], y_vals[:, 1], marker="o", label=model_name)
                ax.set_xlabel("Times")
                ax.set_ylabel("Scores")
        else:
            for model_name in results.keys():
                x_vals = sorted(results[model_name].keys())
                y_vals = [results[model_name][x] for x in x_vals]
                ax.plot(x_vals, y_vals, marker="o", label=model_name)
                ax.set_xlabel("Number of Training Examples")
                ax.set_ylabel(legend)

        ax.legend()
        ax.grid(True)
        return ax


    multi_plot(
        res,
        plot_one,
        mp_nrows=1,
        mp_ncols=3,
        legends=["Scores", "Times", "Efficiency"],
        mp_figsize=(14, 10),
    )
    plt.show()
    pass
