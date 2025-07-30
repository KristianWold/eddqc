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


def collect_raw_data():
    path = pathlib.Path("../integrable_transition/bookkeeping/figure3")
    Ls = [5, 10, 20, 30, 40, 50]
    csr_result_raw = {}
    for L in Ls:
        name = path / f"csr_raw_L={L}.txt"
        csr_result_raw[L] = read_data(name)
    return csr_result_raw


def collect_data():
    path = pathlib.Path("../integrable_transition/bookkeeping/figure3")
    Ls = [5, 10, 20, 30, 40, 50]
    csr_result = {}
    for L in Ls:
        name = path / f"csr_heatmap_blurred_L={L}.txt"
        csr_result[L] = read_data(name)

    scatter_result = {}
    name = path / f"scatterplot_mean.txt"
    with open(name, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert len(row) == 1
            data = list(map(float, row[0].split(" ")))
            scatter_result[Ls[i]] = np.asarray(data)

    scatter_std_result = {}
    name = path / f"scatterplot_std.txt"
    with open(name, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert len(row) == 1
            data = list(map(float, row[0].split(" ")))
            scatter_std_result[Ls[i]] = np.asarray(data)

    name = path / f"scatterplot_theory.txt"  # AI (chaotic) and FF (integrable)
    names = ["AI", "FF"]
    with open(name, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert len(row) == 1
            data = list(map(float, row[0].split(" ")))
            scatter_result[names[i]] = np.asarray(data)

    dist_result = {}
    for key in ["radial", "angular"]:
        for L in [5, 20, 50]:
            name = path / f"{key}_L={L}.txt"
            dist_result[f"{key}_L={L}_exp"] = read_data(name)
            name = path / f"{key}_std_L={L}.txt"
            dist_result[f"{key}_L={L}_exp_std"] = read_data(name)

        name = path / f"{key}_FF_theory.txt"
        dist_result[f"{key}_FF_theory"] = read_data(name)
        name = path / f"{key}_AI_theory.txt"
        dist_result[f"{key}_AI_theory"] = read_data(name)
    return csr_result, scatter_result, scatter_std_result, dist_result


@article_style()
def plot_all(save=False):
    fig = fig_in_a4(1, 0.33, dpi=200)

    fig.subplots_adjust(
        hspace=0.6, wspace=0.5, top=1.06, bottom=0.1, left=0.03, right=0.96
    )
    gs = fig.add_gridspec(100, 100)

    csr_result, scatter_result, scatter_std_result, dist_result = collect_data()

    ag = AxesGroup(6, 6, figs=fig, init=False)

    axes = []
    dx = 16
    interval = 3
    for i in range(6):
        ax = fig.add_subplot(gs[:50, interval + i * dx : (i + 1) * dx])
        axes.append(ax)
    ag.init(ag.figs, axes)
    cax = fig.add_subplot(gs[15:35, 97:98])

    y0 = 51
    ax_scatter = fig.add_subplot(gs[y0:97, 5:39])
    ag_scatter = AxesGroup(1, 1, figs=fig, init=False)
    ag_scatter.init(ag_scatter.figs, [ax_scatter])

    dist_axes = []
    dx = 19
    dy = 20
    interval = 3
    interval_y = 10
    for i in range(3):
        ax = fig.add_subplot(
            gs[y0 : y0 + dy, 43 + interval + i * dx : 43 + (i + 1) * dx]
        )
        dist_axes.append(ax)
    for i in range(3):
        ax = fig.add_subplot(
            gs[
                y0 + interval_y + dy : y0 + interval_y + dy * 2,
                43 + interval + i * dx : 43 + (i + 1) * dx,
            ]
        )
        dist_axes.append(ax)
    ag_dist = AxesGroup(6, 3, figs=fig, init=False)
    ag_dist.init(ag_dist.figs, dist_axes)

    # plot 2d
    xs = ys = [-1, 0, 1]
    vmin = 0
    vmax = 0.6
    cmap = "RdBu_r"

    for i, (L, data) in enumerate(csr_result.items()):
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
        ax.set_title(f"$T$ = {L}", fontsize=FONTSIZE+1)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label=r"probability",
    )
    cax.set_yticks([vmin, vmax])

    ag.set_xlabel("Re ($z$)", fontsize=FONTSIZE+1, labelpad=0.5)
    ag.set_ylabel("Im ($z$)", fontsize=FONTSIZE+1, sharey=1)

    ag.set_xticks(xs, fontsize=FONTSIZE+1, sharex=1)
    ag.set_yticks(ys, fontsize=FONTSIZE+1, sharey=1)

    ag.tick_params(direction="out")
    ag.grid(False)

    # plot scatter
    ax = ag_scatter.axes[0]
    xs = []
    ys = []
    labels = []

    x, y = 0.617, 0.09
    dx = 0.05
    dy = 0.11
    ax.annotate(
        "", xytext=(x, y), xy=(x + dx, y), arrowprops=dict(arrowstyle="->", lw=0.8)
    )
    ax.annotate(
        "",
        xytext=(x + dx / 2, y - dy),
        xy=(x + dx / 2, y + dy),
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.scatter([x + dx / 2], [y], s=1700, ec="k", fc=(1, 1, 1, 1))
    r = 0.065
    theta = np.pi / 2 * 0.9
    ax.annotate(
        "",
        xytext=(x + dx / 2 - 0.001, y - 0.004),
        xy=(x + dx / 2 + r * np.cos(theta), y + r * np.sin(theta)),
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.text(0.652, 0.16, "$r$", fontsize=FONTSIZE + 1, ha="left", va="center")
    ax.text(0.6453, 0.087, r"$\theta$", fontsize=FONTSIZE + 1, ha="left", va="bottom")

    k = "FF"
    v = scatter_result[k]
    x, y = v
    plot1d(
        [x],
        [y],
        ls="",
        marker="*",
        ax=ax,
        ms=MS + 2,
        mew=MEW,
        lw=LW,
        zorder=np.inf,
    )
    labels.append("FF${}_{4,1}$")
    xs.append(x)
    ys.append(y)

    for L in [5, 10, 20, 30, 40, 50]:
        x, y = scatter_result[L]
        std1, std2 = scatter_std_result[L]

        #plot1d(
        #    [x],
        #    [y],
        #    [std1],
        #    [std2],
        #    ls="",
        #    marker="o",
        #    ax=ax,
        #    ms=MS,
        #    capsize=CAPSIZE,
        #    mew=MEW,
        #    lw=LW,
        #    zorder=np.inf,
        #)
        ax.errorbar(
        [x],                   # x-coordinate(s)
        [y],                   # y-coordinate(s)
        xerr=[std1],
        yerr=[std2],            # vertical error bar(s)
        linestyle="",          # no connecting line between points
        marker="o",            # draw each point as a circle
        markersize=MS,         # size of the circle marker
        capsize=CAPSIZE,       # width of the little “caps” on the error bars
        markeredgewidth=MEW,   # thickness of the circle’s edge
        linewidth=LW,          # thickness of the error‐bar lines
        zorder=np.inf,         # draw this on top of everything else
         )
        labels.append(f"$T$ = {L}")
        xs.append(x)
        ys.append(y)

    k = "AI"
    v = scatter_result[k]
    x, y = v
    plot1d(
        [x],
        [y],
        ls="",
        marker="*",
        ax=ax,
        ms=MS + 2,
        mew=MEW,
        lw=LW,
        zorder=np.inf,
        color="k",
    )
    labels.append("AI${}_{4,1}$")
    xs.append(x)
    ys.append(y)

    index = np.argsort(xs)
    xs = np.array(xs)[index]
    ys = np.array(ys)[index]
    plot1d(xs, ys, ls="--", color="k", ax=ax, lw=0.7, zorder=-np.inf)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if i in [0, 1, 2, 4, 6]:
            ax.text(
                x - 0.003,
                y - 0.027,
                labels[i],
                fontsize=7,
                ha="left",
                va="top",
            )
        elif i in [3, 5]:
            ax.text(
                x - 0.001,
                y - 0.027,
                labels[i],
                fontsize=7,
                ha="left",
                va="top",
            )
        elif i in [7]:
            ax.text(
                x + 0.0017,
                y - 0.002,
                labels[i],
                fontsize=7,
                ha="left",
                va="top",
            )

    ag_scatter.set_xlabel(r"$\mathsf{\langle}r\mathsf{\rangle}$", fontsize=FONTSIZE+1)
    ag_scatter.set_ylabel(
        r"$\mathsf{\langle}-{\rm cos(\theta)}\mathsf{\rangle}$", fontsize=FONTSIZE+1
    )
    xs = np.arange(0.62, 0.741, 0.04)
    ys = [-0.2, 0, 0.2]
    ag_scatter.set_xticks(xs, xlim=xs, fontsize=FONTSIZE, sharex=1, xlim_pad_ratio=0.05)
    ag_scatter.set_yticks(ys, ylim=ys, fontsize=FONTSIZE, sharey=1, ylim_pad_ratio=0.05)
    ag_scatter.tick_params(direction="out")
    ag_scatter.grid(False)

    # plot dist 1row
    for i, L in enumerate([5, 20, 50]):
        ax = ag_dist.axes[i]
        x, y = dist_result[f"radial_L={L}_exp"].T
        std = dist_result[f"radial_L={L}_exp_std"].T[0]
        if L == 5:
            color = "C1"
        elif L == 20:
            color = "C3"
        elif L == 50:
            color = "C6"
        ax.set_title(f"$T$ = {L}", fontsize=FONTSIZE+1)
        plot1d(
            x,
            y,
            # std,
            ax=ax,
            ls="",
            lw=LW,
            ms=MS - 2,
            marker="o",
            hollow=0,
            mew=0.6,
            label=f"Exp.",
            color=color,
            zorder=np.inf,
        )
        plot1d(
            x,
            y,
            std,
            ax=ax,
            ls="",
            lw=LW,
            ms=MS,
            errorbar=False,
            mew=0.6,
            color=color,
            fill_alpha=0.4,
        )
        # ax.set_title(f"$T$ = {L}", fontsize=FONTSIZE)

        x, y = dist_result[f"radial_FF_theory"].T
        plot1d(x, y, ax=ax, lw=LW - 0.5, label="FF${}_{4,1}$", color="C0")

        x, y = dist_result[f"radial_AI_theory"].T
        plot1d(x, y, ax=ax, lw=LW - 0.5, label="AI${}_{4,1}$", color="k")
        format_legend(ax, size=5)
        ax.legend(
            labelspacing=0.1,
            framealpha=0.4,
            prop={"size": 5},
            loc="upper left",
        )

    ag_dist.set_xlabel(
        r"$r$", fontsize=FONTSIZE+2, axes=[0, 1, 2], sharex=1, labelpad=0.5
    )
    ag_dist.set_ylabel(r"$P(r)$", fontsize=FONTSIZE+1, axes=[0, 1, 2], sharey=1)
    xs = [0, 0.5, 1]
    ys = [0.0, 1.5, 3.0]
    ag_dist.set_xticks(
        xs, xlim=xs, fontsize=FONTSIZE+1, sharex=1, xlim_pad_ratio=0.05, axes=[0, 1, 2]
    )
    ag_dist.set_yticks(
        ys, ylim=ys, fontsize=FONTSIZE+1, sharey=1, ylim_pad_ratio=0.05, axes=[0, 1, 2]
    )

    # plot dist 2row
    for i, L in enumerate([5, 20, 50]):
        ax = ag_dist.axes[i + 3]
        x, y = dist_result[f"angular_L={L}_exp"].T
        std = dist_result[f"angular_L={L}_exp_std"].T[0]
        if L == 5:
            color = "C1"
        elif L == 20:
            color = "C3"
        elif L == 50:
            color = "C6"
        plot1d(
            x,
            y,
            # std,
            ax=ax,
            ls="",
            lw=LW,
            ms=MS - 2,
            marker="o",
            hollow=0,
            mew=0.6,
            color=color,
            zorder=np.inf,
        )
        plot1d(
            x,
            y,
            std,
            ax=ax,
            ls="",
            lw=LW,
            ms=MS,
            errorbar=False,
            mew=0.6,
            color=color,
            fill_alpha=0.4,
        )

        x, y = dist_result[f"angular_FF_theory"].T
        plot1d(x, y, ax=ax, lw=LW - 0.5, color="C0")

        x, y = dist_result[f"angular_AI_theory"].T
        plot1d(x, y, ax=ax, lw=LW - 0.5, color="k")

    ag_dist.set_xlabel(
        r"$\theta$", fontsize=FONTSIZE+1, axes=[3, 4, 5], sharex=1, labelpad=0.5
    )
    ag_dist.set_ylabel(r"$P(\theta)$", fontsize=FONTSIZE+1, axes=[3, 4, 5], sharey=1)
    xs = [-np.pi, 0.0, np.pi]
    ys = [0, 0.2, 0.4]
    ag_dist.set_xticks(
        xs,
        xlim=xs,
        labels=["$-\pi$", 0, "$\pi$"],
        fontsize=FONTSIZE+1,
        sharex=1,
        xlim_pad_ratio=0.05,
        axes=[3, 4, 5],
    )
    ag_dist.set_yticks(
        ys, ylim=ys, fontsize=FONTSIZE+1, sharey=1, ylim_pad_ratio=0.05, axes=[3, 4, 5]
    )

    ag_dist.tick_params(direction="out")
    ag_dist.grid(False)

    # ----- general -----
    annot_alphabet(
        [ag.axes[0], ax_scatter, ag_dist.axes[0]],
        dx=-0.05,
        dy=0.02,
        fontsize=FONTSIZE+1,
        transform="fig",
        zorder=np.inf,
        left_bond_dict={"b": "a"},
        top_bond_dict={"c": "b"},
    )

    if save:
        fig_name = "figures/fig3_v4.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print("Saved ", fig_name)
