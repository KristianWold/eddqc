from functools import partial
import pathlib
import re

import csv
import h5py
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import (
    Arc,
    BoxStyle,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Path,
    PathPatch,
    Polygon,
    Rectangle,
)
import matplotlib.pyplot as plt
import numpy as np

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4
from plot_toolbox import plot1d as _plot1d
from plot_toolbox import plot2d as _plot2d
from styles import (
    CAPSIZE,
    DASHES,
    FONTSIZE,
    LEGEND_SIZE,
    LW,
    MEW,
    MS,
    USE_TEX,
    article_style,
    format_legend,
)

plot1d = partial(_plot1d, constrained_layout=False)
plot2d = partial(_plot2d, constrained_layout=False)

Kristian_DATA_DIR = pathlib.Path(r"D:\users\ztzhu\exps\quantum_chaos\from")


def read_data(path):
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data = []
        for i, row in enumerate(reader):
            assert len(row) == 1
            data.append(list(map(float, row[0].split(" "))))
        return np.asarray(data)


def collect_data():
    path = pathlib.Path("../../integrable_nonintegrable/book_keeping/figure2")
    types = ["Integrable", "Nonintegrable"]
    result = {}
    for type in types:
        if type == "Integrable":
            name = path / "hist_integrable_L=5.txt"
        else:
            name = path / "hist_nonintegrable_L=10.txt"
        result[type] = read_data(name)  # (200, 200)
    return result


@article_style()
def plot_all(save=False):
    fig = fig_in_a4("1col", 0.2, dpi=200)

    fig.subplots_adjust(
        hspace=0.6, wspace=0.5, top=0.98, bottom=0.04, left=0.05, right=0.96
    )
    gs = fig.add_gridspec(100, 100)

    result = collect_data()

    dy = 35
    ax_circuit0 = fig.add_subplot(gs[:51, :50])
    ax_circuit1 = fig.add_subplot(gs[53:, :50])
    axes = [
        fig.add_subplot(gs[4 : 4 + dy, 60:95]),
        fig.add_subplot(gs[55 : 55 + dy, 60:95]),
    ]

    ag = AxesGroup(2, 1, figs=fig, init=False)

    ag.init(ag.figs, axes)
    cax = fig.add_subplot(gs[40:60, 91:93])

    # plot (a)
    plot_circuit0_a(ax_circuit0)
    plot_circuit0_b(ax_circuit0)

    # plot (b)
    plot_circuit1_a(ax_circuit1)
    plot_circuit1_b(ax_circuit1)

    # ax_circuit0.set_title(f"Free-Fermion circuit", fontsize=FONTSIZE, pad=0.5)
    ax_circuit0.text(
        0.5,
        5.6,
        f"Free-Fermion circuit",
        fontdict={"fontsize": FONTSIZE},
        ha="center",
        va="center",
    )
    ax_circuit1.text(
        0.5,
        5.5,
        f"Chaotic circuit",
        fontdict={"fontsize": FONTSIZE},
        ha="center",
        va="center",
    )

    # plot 2d
    xs = ys = [-1, 0, 1]
    vmin = 0
    vmax = 0.6
    cmap = "RdBu_r"
    for i, (type, data) in enumerate(result.items()):
        ax = ag.axes[i]
        plot2d(
            data,
            interp="none",
            ax=ax,
            plot_cbar=False,
            vmin=vmin,
            vmax=vmax,
            xlim=xs,
            ylim=ys,
            cmap=cmap,
        )
        ax.set_aspect("equal")
        if type == "Integrable":
            title = "Free-Fermion ($T=5$)"
        else:
            title = "Chaotic ($T=10$)"
        ax.set_title(title, fontsize=FONTSIZE - 1, pad=3)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label=r"density",
        pad=0,
    )
    cax.set_yticks([vmin, vmax])
    cax.tick_params("y", length=2, which="major")

    ag.set_xlabel("Re ($z$)", fontsize=FONTSIZE, labelpad=0.5)
    ag.set_ylabel("Im ($z$)", fontsize=FONTSIZE, sharey=1)

    ag.set_xticks(xs, fontsize=FONTSIZE, sharex=1)
    ag.set_yticks(ys, fontsize=FONTSIZE, sharey=1)

    ag.tick_params(direction="out")
    ag.grid(False)

    # ----- general -----
    annot_alphabet(
        [ax_circuit0, ax_circuit1] + axes,
        dx=-0.08,
        dy=0.02,
        fontsize=FONTSIZE,
        transform="fig",
        zorder=np.inf,
        left_bond_dict={"a": "b", "d": "c"},
        top_bond_dict={"a": "c", "d": "b"},
        dx_dict={"b": 0.06},
        dy_dict={"b": -0.03, "d": -0.03},
    )

    if save:
        fig_name = "fig2.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print("Saved ", fig_name)


def plot_circuit0_a(ax):
    total_layer = 4
    numq = 5
    dy = 1
    circle_size = 18
    circuit_start = 0.03
    gate_width = 0.07
    gate_height = 0.7
    circuit_end = 0.6
    gate_start = 0.10
    gate_end = circuit_end - gate_width / 2 - (circuit_start - gate_width / 2) - 0.04
    L = gate_end - gate_start
    interval = L / (total_layer - 1) - gate_width

    def get_x(layer):
        return gate_start + layer * (interval + gate_width)

    lines = {}
    for i in range(1, numq + 1, 1):
        lines[i] = (numq - i) * dy
    for i in range(1, numq + 1, 1):
        y = lines[i]
        ax.plot(
            [circuit_start, circuit_end], [y, y], color="#7f7f7f", lw=1, zorder=-np.inf
        )

    Z_color = "#b8c5e9"
    Y_color = "#95ccb5"
    tq_color = "#f9bfb0"

    def plot_gate(layer, index, color, custom_xy=None, width=None, height=None):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]

        if width is None:
            width = gate_width
        if height is None:
            height = gate_height
        xy = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=BoxStyle.Round(pad=0, rounding_size=0.0),
            facecolor=color,
            zorder=np.inf,
            lw=0.5,
        )
        ax.add_patch(box)

    for i in range(5):
        plot_gate(0, i + 1, Z_color)

    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(1), np.mean([lines[1], lines[2]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(1), np.mean([lines[3], lines[4]])),
        height=gate_height * 2.4,
    )

    for i in range(5):
        plot_gate(2, i + 1, Z_color)

    plot_gate(
        3,
        2,
        tq_color,
        custom_xy=(get_x(3), np.mean([lines[2], lines[3]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        3,
        2,
        tq_color,
        custom_xy=(get_x(3), np.mean([lines[4], lines[5]])),
        height=gate_height * 2.4,
    )

    for i in range(5):
        y = lines[i + 1]
        dx = 0.02
        x0 = circuit_end + 0.03
        ax.scatter([x0, x0 + dx, x0 + dx * 2], [y] * 3, s=0.3, ec="k", fc="k")

    # legend
    fd = {"fontsize": FONTSIZE - 1}
    x = 0.1
    y = 6.58
    plot_gate(
        0,
        0,
        Y_color,
        custom_xy=(x, y),
        width=gate_width * 0.8,
        height=gate_height * 0.8,
    )
    ax.text(
        x + 0.08,
        y + 0.3,
        r"$R_y$",
        ha="center",
        va="top",
        zorder=np.inf,
        fontdict=fd,
    )

    x = 0.35
    plot_gate(
        0,
        0,
        Z_color,
        custom_xy=(x, y),
        width=gate_width * 0.8,
        height=gate_height * 0.8,
    )
    ax.text(
        x + 0.08,
        y + 0.3,
        r"$R_z$",
        ha="center",
        va="top",
        zorder=np.inf,
        fontdict=fd,
    )

    x = 0.6
    plot_gate(
        0,
        0,
        tq_color,
        custom_xy=(x, y),
        width=gate_width * 0.8,
        height=gate_height * 0.8,
    )
    ax.text(
        x + 0.15,
        y + 0.33,
        r"$\sqrt{\rm iSWAP}$",
        ha="center",
        va="top",
        zorder=np.inf,
        fontdict=fd,
    )

    ax.vlines(np.mean([get_x(1), get_x(2)]), -0.5, 5, ls="--", color="k", lw=0.8)
    ax.vlines(0.584, -0.5, 5, ls="--", color="k", lw=0.8)
    ax.vlines(0.709, -0.5, 5, ls="--", color="k", lw=0.8)

    fd = {"fontsize": FONTSIZE - 2}
    ax.text(
        np.mean([get_x(0), get_x(1)]),
        4.83,
        "Layer 1",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )
    ax.text(
        np.mean([get_x(2), get_x(3)]),
        4.83,
        "Layer 2",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )


def plot_circuit0_b(ax):
    total_layer = 2
    numq = 5
    dy = 1
    circle_size = 18
    circuit_start = 0.7
    gate_width = 0.07
    gate_height = 0.7
    circuit_end = 0.9
    gate_start = circuit_start + 0.06
    gate_end = circuit_end - gate_width / 2 - (0.08 - gate_width / 2) + 0.03
    L = gate_end - gate_start
    interval = L / (total_layer - 1) - gate_width

    def get_x(layer):
        return gate_start + layer * (interval + gate_width)

    ax.axis("off")
    lines = {}
    for i in range(1, numq + 1, 1):
        lines[i] = (numq - i) * dy
    for i in range(1, numq + 1, 1):
        y = lines[i]
        ax.plot(
            [circuit_start, circuit_end], [y, y], color="#7f7f7f", lw=1, zorder=-np.inf
        )

    Z_color = "#b8c5e9"
    Y_color = "#95ccb5"
    tq_color = "#f9bfb0"

    def plot_gate(layer, index, color, custom_xy=None, width=None, height=None):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]

        if width is None:
            width = gate_width
        if height is None:
            height = gate_height
        xy = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=BoxStyle.Round(pad=0, rounding_size=0.0),
            facecolor=color,
            zorder=np.inf,
            lw=0.5,
        )
        ax.add_patch(box)

    for i in range(5):
        plot_gate(0, i + 1, Z_color)

    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(1), np.mean([lines[2], lines[3]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(1), np.mean([lines[4], lines[5]])),
        height=gate_height * 2.4,
    )

    fd = {"fontsize": FONTSIZE - 2}
    ax.text(
        np.mean([get_x(0), get_x(1)]),
        4.83,
        "Layer $T$",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )

    plot_trash_can(ax, x0=0.95, y0=0.1, dx=0.1, dy=0.6, dx2=0.026, dx3=0.015, dy2=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 7)


def plot_circuit1_a(ax):
    total_layer = 6
    numq = 5
    dy = 1
    circle_size = 18
    circuit_start = 0.02
    gate_width = 0.05
    gate_height = 0.6
    circuit_end = 0.55
    gate_start = 0.08
    gate_end = circuit_end - gate_width / 2 - (circuit_start - gate_width / 2) - 0.03
    L = gate_end - gate_start
    interval = L / (total_layer - 1) - gate_width

    def get_x(layer):
        return gate_start + layer * (interval + gate_width)

    lines = {}
    for i in range(1, numq + 1, 1):
        lines[i] = (numq - i) * dy
    for i in range(1, numq + 1, 1):
        y = lines[i]
        ax.plot(
            [circuit_start, circuit_end], [y, y], color="#7f7f7f", lw=1, zorder=-np.inf
        )

    Z_color = "#b8c5e9"
    Y_color = "#95ccb5"
    tq_color = "#f9bfb0"

    def plot_gate(layer, index, color, custom_xy=None, width=None, height=None):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]

        if width is None:
            width = gate_width
        if height is None:
            height = gate_height
        xy = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=BoxStyle.Round(pad=0, rounding_size=0.0),
            facecolor=color,
            zorder=np.inf,
            lw=0.5,
        )
        ax.add_patch(box)

    for i in range(5):
        plot_gate(0, i + 1, Y_color)
    for i in range(5):
        plot_gate(1, i + 1, Z_color)

    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(2), np.mean([lines[1], lines[2]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(2), np.mean([lines[3], lines[4]])),
        height=gate_height * 2.4,
    )

    for i in range(5):
        plot_gate(3, i + 1, Y_color)
    for i in range(5):
        plot_gate(4, i + 1, Z_color)

    plot_gate(
        3,
        2,
        tq_color,
        custom_xy=(get_x(5), np.mean([lines[2], lines[3]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        3,
        2,
        tq_color,
        custom_xy=(get_x(5), np.mean([lines[4], lines[5]])),
        height=gate_height * 2.4,
    )

    for i in range(5):
        y = lines[i + 1]
        dx = 0.02
        x0 = circuit_end + 0.03
        ax.scatter([x0, x0 + dx, x0 + dx * 2], [y] * 3, s=0.3, ec="k", fc="k")

    ax.vlines(np.mean([get_x(2), get_x(3)]), -0.5, 5, ls="--", color="k", lw=0.8)
    ax.vlines(0.537, -0.5, 5, ls="--", color="k", lw=0.8)

    fd = {"fontsize": FONTSIZE - 2}
    ax.text(
        np.mean([get_x(0), get_x(2)]),
        4.83,
        "Layer 1",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )
    ax.text(
        np.mean([get_x(3), get_x(5)]),
        4.83,
        "Layer 2",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )


def plot_circuit1_b(ax):
    total_layer = 2
    numq = 5
    dy = 1
    circle_size = 18
    circuit_start = 0.65
    gate_width = 0.05
    gate_height = 0.6
    circuit_end = 0.9
    gate_start = circuit_start + 0.05
    gate_end = circuit_end - gate_width / 2 - (0.08 - gate_width / 2) - 0.04
    L = gate_end - gate_start
    interval = L / (total_layer - 1) - gate_width

    def get_x(layer):
        return gate_start + layer * (interval + gate_width)

    ax.axis("off")
    lines = {}
    for i in range(1, numq + 1, 1):
        lines[i] = (numq - i) * dy
    for i in range(1, numq + 1, 1):
        y = lines[i]
        ax.plot(
            [circuit_start, circuit_end], [y, y], color="#7f7f7f", lw=1, zorder=-np.inf
        )

    Z_color = "#b8c5e9"
    Y_color = "#95ccb5"
    tq_color = "#f9bfb0"

    def plot_gate(layer, index, color, custom_xy=None, width=None, height=None):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]

        if width is None:
            width = gate_width
        if height is None:
            height = gate_height
        xy = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=BoxStyle.Round(pad=0, rounding_size=0.0),
            facecolor=color,
            zorder=np.inf,
            lw=0.5,
        )
        ax.add_patch(box)

    for i in range(5):
        plot_gate(0, i + 1, Y_color)
    for i in range(5):
        plot_gate(1, i + 1, Z_color)

    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(2), np.mean([lines[2], lines[3]])),
        height=gate_height * 2.4,
    )
    plot_gate(
        1,
        2,
        tq_color,
        custom_xy=(get_x(2), np.mean([lines[4], lines[5]])),
        height=gate_height * 2.4,
    )

    plot_trash_can(ax, x0=0.95, y0=0.1, dx=0.1, dy=0.6, dx2=0.026, dx3=0.015, dy2=0.55)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 5.8)

    ax.vlines(0.66, -0.5, 5, ls="--", color="k", lw=0.8)

    fd = {"fontsize": FONTSIZE - 2}
    ax.text(
        np.mean([get_x(0), get_x(2)]),
        4.83,
        "Layer $T$",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )


def plot_trash_can(ax, x0, y0, dx, dy, dx2, dx3, dy2):
    ellipse = Ellipse(
        (x0, y0), width=dx / 2, height=dy / 2, ec="k", fc="gray", lw=0.6, zorder=np.inf
    )
    ax.add_patch(ellipse)

    x = [x0 - dx2, x0 - dx3, x0 + dx3, x0 + dx2]
    y = [y0, y0 - dy2, y0 - dy2, y0]
    ax.add_patch(Polygon(xy=list(zip(x, y)), fill=False, zorder=4000, lw=0.6))
