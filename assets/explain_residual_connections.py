import os
from enum import Enum

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

MPL_FONT_SIZE_TITLE: int = 16
MPL_FONT_SIZE_LABEL: int = 14
MPL_FONT_SIZE_TICKS: int = 14


class PlotMode(Enum):
    SINGLE = 1
    STACKED_VERTICALLY = 2
    STACKED_HORIZONTALLY = 3


PLOT_MODE: PlotMode = PlotMode.SINGLE


def solve_analytically(n_steps: int) -> list[float]:
    xs = [2**t for t in range(n_steps + 1)]
    return xs


def solve_without_residual_connections(x_curr: float) -> float:
    x_next = 2 * x_curr
    return x_next


def solve_with_residual_connections(x_curr: float) -> float:
    residual = x_curr
    x_next = x_curr + residual
    return x_next


def plot_residual_connections_explaination(
    ax1: Axes,
    ax2: Axes,
    ax3: Axes,
    xs_solved_analytically: list[float],
    xs_solved_without_residual_connection: list[float],
    xs_solved_with_residual_connection: list[float],
    plot_mode: PlotMode,
) -> None:
    """
    This function plots the same function in three different ways: analytical solution, recursive solution without
    residual connections, recursive solution with residual connections.
    """
    assert len(xs_solved_analytically) == len(xs_solved_without_residual_connection)
    assert len(xs_solved_analytically) == len(xs_solved_with_residual_connection)
    ts = list(range(len(xs_solved_analytically)))
    axs = [ax1, ax2, ax3]
    xs = [
        xs_solved_analytically,
        xs_solved_without_residual_connection,
        xs_solved_with_residual_connection,
    ]
    labels = [
        "analytical solution",
        "without residual connection",
        "with residual connection",
    ]
    colors = ["blue", "green", "red"]
    for idx, (ax_curr, xs_curr, label_curr, color_curr) in enumerate(
        zip(axs, xs, labels, colors)
    ):
        ax_curr.plot(
            ts,
            xs_curr,
            "*-",
            color=color_curr,
        )
        ax_curr.grid(True)
        ax_curr.set_xlim(0, len(ts) - 1)
        ax_curr.set_ylim(0, max(xs_curr) + 1)
        ax_curr.tick_params(labelsize=MPL_FONT_SIZE_TICKS)
        # ax_curr.set_title(
        #     label_curr,
        #     fontdict={"fontsize": MPL_FONT_SIZE_TITLE, "fontweight": "bold"},
        # )
        if (plot_mode is not PlotMode.STACKED_VERTICALLY) or (
            plot_mode == PlotMode.STACKED_VERTICALLY and idx == 2
        ):
            ax_curr.set_xlabel(
                r"$t$",
                fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
            )
        if (plot_mode is not PlotMode.STACKED_HORIZONTALLY) or (
            plot_mode == PlotMode.STACKED_HORIZONTALLY and idx == 0
        ):
            ax_curr.set_ylabel(
                r"$x$",
                fontdict={"fontsize": MPL_FONT_SIZE_LABEL, "fontweight": "bold"},
                rotation=0,
                labelpad=20,
            )
        if plot_mode == PlotMode.SINGLE and idx == 0:
            break


if __name__ == "__main__":
    # Fill the list of values using three different approaches.
    x_start = 1.0
    n_steps = 5
    xs_solved_analytically: list[float] = solve_analytically(n_steps)
    xs_solved_without_residual_connection: list[float] = [x_start]
    for _ in range(n_steps):
        xs_solved_without_residual_connection.append(
            solve_without_residual_connections(
                xs_solved_without_residual_connection[-1]
            )
        )
    xs_solved_with_residual_connection: list[float] = [x_start]
    for _ in range(n_steps):
        xs_solved_with_residual_connection.append(
            solve_with_residual_connections(xs_solved_with_residual_connection[-1])
        )
    # Plot them, should look the same.
    if PLOT_MODE == PlotMode.SINGLE:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = ax3 = ax1
    elif PLOT_MODE == PlotMode.STACKED_VERTICALLY:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    elif PLOT_MODE == PlotMode.STACKED_HORIZONTALLY:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    else:
        raise ValueError
    plot_residual_connections_explaination(
        ax1,
        ax2,
        ax3,
        xs_solved_analytically,
        xs_solved_without_residual_connection,
        xs_solved_with_residual_connection,
        PLOT_MODE,
    )
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plot_residual_connections.pdf"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print("Plot written to '%s'" % filepath)
    plt.show()
