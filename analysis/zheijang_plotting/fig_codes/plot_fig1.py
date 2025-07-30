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
    path = pathlib.Path("../../theoretic_AI_and_FF/book_keeping/figure1")
    types = ["AI_inf", "AI_41", "FF_inf", "FF_41"]
    result = {}
    for type in types:
        name = path / f"hist_{type.replace('_', '')}_theory.txt"
        result[type] = read_data(name)  # (200, 200)
    return result


@article_style()
def plot_all(save=False):
    fig = fig_in_a4(1, 0.33, dpi=200)

    fig.subplots_adjust(
        hspace=0.6, wspace=0.5, top=0.95, bottom=0.07, left=0.05, right=1.00
    )
    gs = fig.add_gridspec(100, 100)

    result = collect_data()

    ax_circuit = fig.add_subplot(gs[:45, :31])
    ax_diagram = fig.add_subplot(gs[54:99, :31])

    ag = AxesGroup(4, 2, figs=fig, init=False)

    dx = 26
    dy = 41
    x0 = 36
    y0 = 1
    interval_x = 4 + dx
    interval_y = 12 + dy
    axes = [
        fig.add_subplot(gs[y0 : y0 + dy, x0 : x0 + dx]),
        fig.add_subplot(gs[y0 : y0 + dy, x0 + interval_x : x0 + interval_x + dx]),
        fig.add_subplot(gs[y0 + interval_y : y0 + interval_y + dy, x0 : x0 + dx]),
        fig.add_subplot(
            gs[
                y0 + interval_y : y0 + interval_y + dy,
                x0 + interval_x : x0 + interval_x + dx,
            ]
        ),
    ]
    ag.init(ag.figs, axes)
    cax = fig.add_subplot(gs[40:60, 91:93])

    # plot (a)
    plot_circuit(ax_circuit)

    # plot (b)
    plot_diagram(ax_diagram)

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
        title = {
            "AI_41": r"AI${}_{4,1}$",
            "FF_41": r"FF${}_{4,1}$",
            "AI_inf": r"AI${}_{\infty,\infty}$",
            "FF_inf": r"FF${}_{\infty,\infty}$",
        }[type]
        ax.set_title(title, fontsize=FONTSIZE+2)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label=r"probability",
    )
    cax.set_yticks([vmin, vmax])

    ag.set_xlabel("Re ($z$)", fontsize=FONTSIZE+2, labelpad=0.5)
    ag.set_ylabel("Im ($z$)", fontsize=FONTSIZE+2, sharey=1)

    ag.set_xticks(xs, fontsize=FONTSIZE+2, sharex=1)
    ag.set_yticks(ys, fontsize=FONTSIZE+2, sharey=1)

    ag.tick_params(direction="out")
    ag.grid(False)

    # ----- general -----
    annot_alphabet(
        [ax_circuit, ax_diagram, ag.axes[0], ag.axes[2], ag.axes[1], ag.axes[3]],
        dx=-0.05,
        dy=0.02,
        fontsize=FONTSIZE+2,
        transform="fig",
        zorder=np.inf,
        left_bond_dict={"b": "a"},
        top_bond_dict={"c": "a", "e": "a", "d": "b", "f": "b"},
        dx_dict={"a": 0.03},
    )

    if save:
        fig_name = "fig1.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print("Saved ", fig_name)


def plot_circuit(ax):
    ax.axis("off")
    dy = 0.1
    x0 = 0.15
    dx = 0.7

    # ancilla qubit
    color = "#b12f53"
    for i in range(3):
        y = 0.9 - i * dy
        ax.plot([x0, x0 + dx], [y] * 2, color=color, lw=1)
        ax.text(
            x0 - 0.05, y, r"$|0\mathsf{\rangle}$", ha="center", va="center", color=color
        )
    y = 0.55
    ax.plot([x0, x0 + dx], [y] * 2, color=color, lw=1)
    ax.text(
        x0 - 0.05, y, r"$|0\mathsf{\rangle}$", ha="center", va="center", color=color
    )
    ax.scatter(
        [x0 + dx * 0.1] * 3 + [x0 + dx * 0.9] * 3,
        [0.58, 0.63, 0.68] * 2,
        s=2,
        color=color,
    )

    # data qubit
    color = "k"
    for i in range(3):
        y = 0.45 - i * dy
        ax.plot([x0, x0 + dx], [y] * 2, color=color, lw=1)
    y = 0.1
    ax.plot([x0, x0 + dx], [y] * 2, color=color, lw=1)
    ax.scatter(
        [x0 + dx * 0.1] * 3 + [x0 + dx * 0.9] * 3,
        [0.13, 0.18, 0.23] * 2,
        s=2,
        color=color,
    )

    # gate
    ax.add_patch(
        Rectangle(
            (0.3, 0.05), width=0.4, height=0.9, zorder=1000, ec="k", color="#6eaadb"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.25, 0.025),
            width=0.5,
            height=0.47,
            zorder=2000,
            ec="k",
            ls="--",
            color="#edccaf",
            alpha=0.7,
        )
    )
    ax.text(
        0.5,
        0.75,
        r"$U$",
        ha="center",
        va="center",
        fontsize=13,
        zorder=np.inf,
        usetex=USE_TEX,
    )
    ax.text(
        0.5,
        0.25,
        r"$\Lambda$",
        ha="center",
        va="center",
        fontsize=13,
        zorder=np.inf,
        usetex=USE_TEX,
    )

    # curly brace
    brace = CurlyBrace(
        x=x0 - 0.06,
        y=0.1,
        width=0.05,
        height=0.35,
        lw=0.8,
        curliness=0.3,
        pointing="left",
        color="k",
    )
    ax.add_artist(brace)
    ax.text(x0 - 0.1, 0.278, r"$\rho$", ha="center", va="center", color="k")

    brace = CurlyBrace(
        x=x0 + 0.71,
        y=0.1,
        width=0.05,
        height=0.35,
        lw=0.8,
        curliness=0.3,
        pointing="right",
        color="k",
    )
    ax.add_artist(brace)
    ax.text(x0 + 0.81, 0.278, r"$\rho\prime$", ha="center", va="center", color="k")

    brace = CurlyBrace(
        x=x0 + 0.71,
        y=0.55,
        width=0.05,
        height=0.35,
        lw=0.8,
        curliness=0.3,
        pointing="right",
        color="k",
    )
    ax.add_artist(brace)
    plot_trash_can(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_diagram(ax):
    ax.axis("off")
    x, y = 0.00, 0.6
    dx = 0.5
    dy = 0.6
    ax.annotate(
        "", xytext=(x, y), xy=(x + dx, y), arrowprops=dict(arrowstyle="->", lw=0.8)
    )
    ax.annotate(
        "",
        xytext=(x + dx / 2, y - dy / 2),
        xy=(x + dx / 2, y + dy / 2),
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.scatter([x + dx / 2], [y], s=1700, ec="k", fc=("#096a68", 0.25))
    ax.text(
        0.28,
        0.886,
        r"Im ($\lambda$)",
        fontsize=FONTSIZE + 1,
        ha="left",
        va="center",
        usetex=USE_TEX,
    )
    ax.text(
        0.395,
        0.53,
        r"Re ($\lambda$)",
        fontsize=FONTSIZE + 1,
        ha="left",
        va="center",
        usetex=USE_TEX,
    )
    ax.text(
        0.09,
        0.19,
        r"$z_j=\frac{\lambda_j-\lambda_j^{\rm NN}}{\lambda_j-\lambda_j^{\rm NNN}}$",
        fontsize=FONTSIZE + 2,
        ha="left",
        va="center",
        usetex=USE_TEX,
    )
    x0, y0, r0 = 0.315, 0.5, 0.04
    x1, y1, r1 = 0.77, 0.5, 0.28

    ax.scatter([x0], [y0], s=80, ec="k", linewidths=0.5, fc=(1, 1, 1, 1))
    ax.scatter(
        [x0 - 0.013, x0 + 0.008, x0 - 0.005],
        [y0 - 0.008, y0 - 0.001, y0 + 0.014],
        s=2,
        ec="k",
        linewidths=0.1,
        fc=("#bf594f", 0.5),
    )


    offset = 0.2
    ax.scatter(
        [x0 - 0.013, x0 + 0.008, x0 - 0.005],
        [y0 + 0.008 + offset, y0 + 0.001 + offset, y0 - 0.014 + offset],
        s=2,
        ec="k",
        linewidths=0.1,
        fc=("#bf594f", 0.5),
    )

    ax.scatter([x1], [y1], s=4000, ec="k", fc=(1, 1, 1, 1))

    rs = np.array([ 0.026, 0.05 , 0.045, 0.073, 0.108, 0.102, 0.044, 0.057, 0.033, 0.07 , 0.085, 0.036, 0.064, 0.107, 0.05 , 0.127, 0.135, 0.104, 0.14 ,
       0.094, 0.058, 0.089, 0.045, 0.128, 0.019, 0.047, 0.102, 0.001, 0.077])  # fmt:skip
    thetas = np.array([ -1.195, 2.589, -1.592, 1.621, 1.698, 0.396, 1.953, -1.135, -1.48, -1.059, -0.221, 2.154, 2.395, 1.925, 2.47, 1.925, 2.694, -0.098, 0.994, 0.144, -2.055, -3.002, 1.523, 0.274, -1.441, -0.572, -2.763, 1.092, -2.226])  # fmt:skip
    
    #conjugate invariant
    rs = np.concatenate([rs[:15], rs[:15]])
    thetas = np.concatenate([thetas[:15], -thetas[:15]])


    
    ax.scatter(
        x + dx / 2 + rs * np.cos(thetas),
        y + rs * np.sin(thetas),
        s=2,
        ec="k",
        linewidths=0.1,
        fc=("#bf594f", 0.5),
    )

    d = x1 - x0
    theta = np.arcsin(r1 / d)
    x = [x0, x1 - (r1 - 0.06) * np.sin(theta)]
    ya = [y0 + r0, y1 + (r1 + 0.01) * np.cos(theta)]
    yb = [y0 - r0, y1 - (r1 + 0.01) * np.cos(theta)]
    plot1d(x, ya, ax=ax, color="k", ls="--", grid=0, lw=0.7)
    plot1d(x, yb, ax=ax, color="k", ls="--", grid=0, lw=0.7)

    xs = [0.65, 0.71, 0.85]
    ys = [0.43, 0.60, 0.455]

    ax.scatter(xs, ys, s=20, ec="k", linewidths=0.5, fc=("#bf594f", 0.5))
    
    ax.annotate(
        "",
        xytext=(xs[0], ys[0]),
        xy=(xs[1], ys[1]),
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    ax.annotate(
        "",
        xytext=(xs[0], ys[0]),
        xy=(xs[2], ys[2]),
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )

    ax.text(
        xs[0],
        ys[0] - 0.03,
        r"$\lambda_j$",
        fontsize=FONTSIZE,
        ha="center",
        va="top",
        usetex=USE_TEX,
    )
    ax.text(
        xs[1] + 0.03,
        ys[1],
        r"$\lambda_j^{\rm NN}$",
        fontsize=FONTSIZE,
        ha="left",
        va="center",
        usetex=USE_TEX,
    )
    ax.text(
        xs[2],
        ys[2] - 0.04,
        r"$\lambda_j^{\rm NNN}$",
        fontsize=FONTSIZE,
        ha="center",
        va="top",
        usetex=USE_TEX,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_trash_can(ax):
    x0, y0 = 0.95, 0.75
    dx = 0.1
    dy = 0.07
    ellipse = Ellipse(
        (x0, y0), width=dx / 2, height=dy / 2, ec="k", fc="gray", lw=0.6, zorder=np.inf
    )
    ax.add_patch(ellipse)

    x = [x0 - 0.025, x0 - 0.015, x0 + 0.015, x0 + 0.025]
    y = [y0, y0 - 0.06, y0 - 0.06, y0]
    ax.add_patch(Polygon(xy=list(zip(x, y)), fill=False, zorder=4000, lw=0.6))


def CurlyBrace(
    x, y, width=1 / 8, height=1.0, curliness=1 / np.e, pointing="left", **patch_kw
):
    """Create a matplotlib patch corresponding to a curly brace (i.e. this thing: "{")

    Parameters
    ----------
    x : float
        x position of left edge of patch
    y : float
        y position of bottom edge of patch
    width : float
        horizontal span of patch
    height : float
        vertical span of patch
    curliness : float
        positive value indicating extent of curliness; default (1/e) tends to look nice
    pointing : str
        direction in which the curly brace points (currently supports 'left' and 'right')
    **patch_kw : any keyword args accepted by matplotlib's Patch

    Returns
    -------
    matplotlib PathPatch corresponding to curly brace

    Notes
    -----
    It is useful to supply the `transform` parameter to specify the coordinate system for the Patch.

    To add to Axes `ax`:
    cb = CurlyBrace(x, y)
    ax.add_artist(cb)

    This has been written as a function that returns a Patch because I saw no use in making it a class, though one could extend matplotlib's Patch as an alternate implementation.

    Thanks to:
    https://graphicdesign.stackexchange.com/questions/86334/inkscape-easy-way-to-create-curly-brace-bracket
    http://www.inkscapeforum.com/viewtopic.php?t=11228
    https://css-tricks.com/svg-path-syntax-illustrated-guide/
    https://matplotlib.org/users/path_tutorial.html

    Ben Deverett, 2018.


    Examples
    --------
    >>>from curly_brace_patch import CurlyBrace
    >>>import matplotlib.pyplot as pl
    >>>fig,ax = pl.subplots()
    >>>brace = CurlyBrace(x=.4, y=.2, width=.2, height=.6, pointing='right', transform=ax.transAxes, color='magenta')
    >>>ax.add_artist(brace)

    """

    verts = np.array(
        [
            [width, 0],
            [0, 0],
            [width, curliness],
            [0, 0.5],
            [width, 1 - curliness],
            [0, 1],
            [width, 1],
        ]
    )

    if pointing == "left":
        pass
    elif pointing == "right":
        verts[:, 0] = width - verts[:, 0]

    verts[:, 1] *= height

    verts[:, 0] += x
    verts[:, 1] += y

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    path = Path(verts, codes)

    # convert `color` parameter to `edgecolor`, since that's the assumed intention
    patch_kw["edgecolor"] = patch_kw.pop("color", "black")

    pp = PathPatch(path, facecolor="none", **patch_kw)
    return pp
