import os

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np


N_VALUES: int = 128  # size of intial random distribution
INIT_SCALE: float = 2.4  # arbitrary scale of intial random distribution
INIT_SHIFT: float = 4.1  # arbitrary shift of initial random distribution
LEARNED_SCALE: float = 0.4  # gamma in paper formula
LEARNED_SHIFT: float = -1.2  # beta in paper formula


def get_initial_embedding(
    n_values: int,
    init_scale: float,
    init_shift: float,
) -> np.ndarray:
    """
    Creates an normal distribution of n values, scales them and shifts them afterwards.
    This simulates an existing embedding.
    """
    xs = np.random.standard_normal(n_values) * init_scale + init_shift
    return xs


def plot_layernorm_explaination(
    ax: Axes,
    embedding_initial: np.ndarray,
):
    """
    This function shows the individual steps of the LayerNorm using a given random distribution.
    """
    embedding_mean_shifted = embedding_initial - embedding_initial.mean()
    embedding_normalized = embedding_mean_shifted / np.sqrt(embedding_initial.var())
    embedding_normalized_scaled = embedding_normalized * LEARNED_SCALE
    embedding_normalized_scaled_shifted = embedding_normalized_scaled + LEARNED_SHIFT

    # We want to see the values of the embedding (random distribution) better, so we make them shift randomly a little bit.
    x_vis_shift = np.random.standard_normal(N_VALUES) * 0.01
    ax.axhline(0, -0.5, 4.5, color="black")
    ax.plot(0 + x_vis_shift, embedding_initial, ".")
    ax.plot(1 + x_vis_shift, embedding_mean_shifted, ".")
    ax.plot(2 + x_vis_shift, embedding_normalized, ".")
    ax.plot(3 + x_vis_shift, embedding_normalized_scaled, ".")
    ax.plot(4 + x_vis_shift, embedding_normalized_scaled_shifted, ".")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-15, 15)  # may change depending on the parameters we use
    # Remove the outer box of the plot, only keep the left y spine.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # The xtick labels are added in the Powerpoint, so drop them here.
    ax.set_xticks([])
    ax.set_yticks([0])


if __name__ == "__main__":
    # Perform the different sub-calculations and store them so we can visualize
    # them individually.
    embedding_initial = get_initial_embedding(N_VALUES, INIT_SCALE, INIT_SHIFT)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_layernorm_explaination(
        ax,
        embedding_initial,
    )
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plot_layernorm.pdf"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print("Plot written to '%s'" % filepath)
    plt.show()
