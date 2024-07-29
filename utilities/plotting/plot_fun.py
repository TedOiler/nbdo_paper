import numpy as np
from matplotlib import pyplot as plt

from utilities.basis.basis import step, b_spline_basis


# plt.style.use('default')
# plt.style.use('bmh')
# plt.style.use('ggplot')
# plt.style.use('grayscale')


def plot_basis(ax, T, w, f, run, size=35) -> None:
    plt.style.use('fivethirtyeight')
    ax.plot(np.asarray(T, float), [f(t, w=w) for t in T], zorder=-1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '.5', '1'])
    ax.set_title(f"({chr(run + 1 + 96)})", fontsize=16)
    # Calculate knots and weights
    knots = [(1 / (len(w) - 1 + 1)) * (i + 1) for i in
             range(len(w) - 1 + 1 - 1)]
    weights = [w[i + 1] for i in range(
        len(w) - 1)]  # we want to exclude the first and last points, as these will be drawn with a different colour

    # Draw knots
    ax.scatter(knots, weights, edgecolor="darkorange", facecolors='darkorange', s=size,
               zorder=1)  # internal knots with empty circles
    ax.scatter([0, 1], [w[0], w[len(w) - 1]], edgecolor="black", facecolors='black', s=size,
               zorder=1)  # support knots with empty circles
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=6)


def subplot_results(sub_x, sub_y, T, results, style='fivethirtyeight', size=35, save=True) -> None:
    if style is not None:
        plt.style.use(style)
    fig, ax = plt.subplots(sub_x, sub_y, figsize=(sub_x * sub_y, sub_x * sub_y), tight_layout=True)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    row_to_plot = 0
    for i in range(sub_x):
        for j in range(sub_y):
            try:
                plot_basis(ax=ax[i, j], T=T, w=results[row_to_plot, :].tolist(), f=step, size=size, run=row_to_plot)
            except IndexError:
                pass
            row_to_plot += 1
    if save:
        fig2 = plt.savefig('./results/myexp.png')
    fig = plt.gcf()
    return fig


def plot_b_spline_basis(ax, T, w, k, run, size=35):
    plt.style.use('fivethirtyeight')
    knots = np.linspace(0, 1, len(w) + k + 1)
    b_spline_vals = [sum(w[i] * b_spline_basis(t, k, i, knots) for i in range(len(w))) for t in T]
    ax.plot(T, b_spline_vals, zorder=-1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '.5', '1'])
    ax.set_title(f"({chr(run + 1 + 96)})", fontsize=16)
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=6)

    # Calculate the y-values at the internal knots
    internal_knots = knots[1:-1]
    knot_vals = [sum(w[i] * b_spline_basis(t, k, i, knots) for i in range(len(w))) for t in internal_knots]

    # Calculate the y-values at the boundary knots (evaluating right at the boundaries might cause issues)
    boundary_knots = [knots[0], knots[-1]]
    boundary_vals = [
        sum(w[i] * b_spline_basis(knots[0] + 1e-9, k, i, knots) for i in range(len(w))),
        sum(w[i] * b_spline_basis(knots[-1] - 1e-9, k, i, knots) for i in range(len(w)))
    ]

    # Draw knots
    ax.scatter(internal_knots, knot_vals, edgecolor="darkorange", facecolors='darkorange', s=size,
               zorder=1)  # internal knots
    ax.scatter([knots[0], knots[-1]], boundary_vals, edgecolor="black", facecolors='black', s=size,
               zorder=1)  # support knots


def subplot_b_spline_results(sub_x, sub_y, T, results, k, style='fivethirtyeight', size=35, show=True, graph_title=None):
    if style is not None:
        plt.style.use(style)
    fig, axs = plt.subplots(sub_x, sub_y, figsize=(sub_x * sub_y, sub_x * sub_y), tight_layout=True)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    row_to_plot = 0
    for i in range(sub_x):
        for j in range(sub_y):
            try:
                plot_b_spline_basis(ax=axs[i, j], T=T, w=results[row_to_plot, :].tolist(), k=k, run=row_to_plot,
                                    size=size)
            except IndexError:
                pass
            row_to_plot += 1
    fig2 = plt.savefig(graph_title)
    if show:
        plt.show()
    return fig
