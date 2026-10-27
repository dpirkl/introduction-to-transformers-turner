import os

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np


EXAMPLE_VALUES: list[float] = [-4.1, 5.3, 0.2, 2.7, -3.2]
VISUALIZATION_X_MIN: float = -4.0
VISUALIZATION_X_MAX: float = 4.0

MPL_FONT_SIZE_LABEL: int = 24
MPL_FONT_SIZE_TICKS: int = 16


def plot_exp(
    ax: Axes,
    x_min: float,
    x_max: float,
) -> None:
    """
    Plots the e^x function.
    This plot will be used in the Powerpoint presentation to quickly remind
    the students that e^x is always greater than 0.
    Will be used to explain the softmax function.
    """
    xs = np.arange(x_min, x_max, 0.01)
    ys = np.exp(xs)
    y_min_vis = -0.1
    y_max_vis = ys.max()
    # Plot the function and mark also the x=0, y=0 coordinate system axes.
    ax.axhline(0, x_min, x_max, color="black")
    ax.axvline(0, y_min_vis, y_max_vis, color="black")
    ax.plot(xs, ys, linewidth=2)
    # Customization.
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_vis, y_max_vis)
    ax.tick_params(labelsize=MPL_FONT_SIZE_TICKS)
    ax.set_xlabel(
        r"$x$",
        fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
    )
    ax.set_ylabel(
        r"$e^x$",
        fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
        rotation=0,
        labelpad=20,
    )


def calculate_softmax(
    xs: list[float],
) -> None:
    """
    Performs the softmax function on the given input sequence.
    Since we use the values in the Powerpoint presentation, it will also
    print out the input values as well as the result values to the console
    for easy copy-paste to the presentation.
    """
    print("Input Values:")
    for x in xs:
        print(x)
    print("Output Values:")
    ys_exp = np.exp(xs)
    ys_softmax = ys_exp / ys_exp.sum()
    for y in ys_softmax:
        print("%.3f" % y)
    # Values should be between 0 and 1 and they should sum up to 1.
    assert 0.0 < ys_softmax.min() < 1.0, "min not between 0 and 1"
    assert 0.0 < ys_softmax.max() < 1.0, "max not between 0 and 1"
    assert np.isclose(ys_softmax.sum(), 1.0), "does not sum up to 1"
    print("sum: %.3f" % ys_softmax.sum())
    print("min: %.3f" % ys_softmax.min())
    print("max: %.3f" % ys_softmax.max())


if __name__ == "__main__":
    # Calculate the softmax function on some example values.
    calculate_softmax(EXAMPLE_VALUES)

    # Plot the e^x function to show that the values are always greater than 0.
    fig_exp, ax_exp = plt.subplots(1, 1, figsize=(12, 8))
    plot_exp(ax_exp, VISUALIZATION_X_MIN, VISUALIZATION_X_MAX)
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plot_softmax.pdf"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print("Plot written to '%s'" % filepath)
    plt.show()
