import os

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np


MPL_FONT_SIZE_LABEL: int = 24
MPL_FONT_SIZE_TICKS: int = 16
MPL_FONT_SIZE_LEGEND: int = 16


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def plot_activation_functions(
    ax: Axes, x_min: float, x_max: float, include_sigmoid: bool
) -> None:
    xs = np.arange(x_min, x_max, 0.01)
    y_min_vis = -1.0
    y_max_vis = x_max
    ax.axhline(0, x_min, x_max, color="black")
    ax.axvline(0, y_min_vis, y_max_vis, color="black")
    ax.plot(xs, relu(xs), linewidth=3, label="ReLU")
    ax.plot(xs, gelu(xs), linewidth=3, label="GELU")
    ax.plot(xs, silu(xs), linewidth=3, label="SiLU")
    if include_sigmoid:
        ax.plot(xs, sigmoid(xs), linewidth=3, label="sigmoid")
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_vis, y_max_vis)
    ax.tick_params(labelsize=MPL_FONT_SIZE_TICKS)
    ax.set_xlabel(
        r"$x$",
        fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
    )
    ax.set_ylabel(
        r"$f(x)$",
        fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
        rotation=0,
        labelpad=30,
    )
    ax.legend(fontsize=MPL_FONT_SIZE_LEGEND)


if __name__ == "__main__":
    for include_sigmoid, (x_min, x_max), filename in zip(
        [False, True],
        [(-5.0, 5.0), (-5.0, 5.0)],
        [
            "plot_activation_functions.pdf",
            "plot_activation_functions_including_sigmoid.pdf",
        ],
    ):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_activation_functions(ax, x_min, x_max, include_sigmoid)
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print("Plot written to '%s'" % filepath)
        plt.show()
