from copy import deepcopy
from itertools import cycle

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from colors import color_palette
from fig_utils import subplot, write_footnote

footnote = write_footnote  # for compatibility


mpl.rcParams.update({"font.size": 14, "lines.linewidth": 2})


def plot1d(
    x,
    y=None,
    yerr=None,
    xsim=None,
    ysim=None,
    xlabel=None,
    ylabel=None,
    xlim="auto",
    ylim="auto",
    title=None,
    fig_title=None,
    grid=True,
    errorbar=True,
    ax=None,
    index_ls_map={},
    figsize=[6.5, 4],
    xtick_unit_pi=False,
    fill_alpha=0.2,
    sim_label=None,
    sim_same_color=False,
    sim_kwargs=None,
    legend_size=11,
    legend_ncol=1,
    hollow=True,
    constrained_layout=True,
    legend_loc="best",
    title_color=None,
    title_size=None,
    **line_kwargs,
):
    if "color" in line_kwargs and line_kwargs["color"] is None:
        line_kwargs.pop("color")
    if ax is None:
        ax = subplot(figsize=figsize)
    fig = ax.figure
    fig.set_constrained_layout(constrained_layout)
    if y is None:
        y = x
        x = np.arange(np.shape(y)[-1])
    x, y = np.asarray(x), np.asarray(y)
    multi_lines = len(y.shape) > 1
    n = y.shape[0] if multi_lines else 1
    sim_kwargs = sim_kwargs or {}

    # record the min and max values of x and y for this axis
    lim_vars = {
        "xmin": x.min(),
        "xmax": x.max(),
        "ymin": y.min(),
        "ymax": y.max(),
    }
    if yerr is not None:
        yerr = np.asarray(yerr)
        lim_vars["ymin"] = min((y - yerr).min(), lim_vars["ymin"])
        lim_vars["ymax"] = max((y + yerr).max(), lim_vars["ymax"])
    if ysim is not None:
        ysim = np.asarray(ysim)
        lim_vars["ymin"] = min(ysim.min(), lim_vars["ymin"])
        lim_vars["ymax"] = max(ysim.max(), lim_vars["ymax"])
    if xsim is not None:
        xsim = np.asarray(xsim)
        lim_vars["xmin"] = min(xsim.min(), lim_vars["xmin"])
        lim_vars["xmax"] = max(xsim.max(), lim_vars["xmax"])
    else:
        xsim = x
    for key in lim_vars:
        full_key = f"_custom_key_{key}"
        v_local = lim_vars[key]
        if hasattr(ax, full_key):
            v = getattr(ax, full_key)
            if "min" in key:
                v = min(v, v_local)
            else:
                v = max(v, v_local)
        else:
            v = v_local
        setattr(ax, full_key, v)
        lim_vars[key] = v

    legend_on = False
    if "marker" in line_kwargs and line_kwargs["marker"] == "o":
        if hollow:
            line_kwargs.setdefault("mfc", "none")
        line_kwargs.setdefault("mew", "1.5")
        line_kwargs.setdefault("ms", "7")

    if not multi_lines:
        if ysim is not None:
            line_kwargs.setdefault("label", "exp.")
            if sim_label is None:
                sim_label = "sim."
            sim_kwargs.setdefault("label", sim_label)
            sim_kwargs.setdefault("ls", "--")
            if not sim_same_color:
                ax.plot(xsim, ysim, **sim_kwargs)
        if yerr is None or not errorbar:
            ax.plot(x, y, **line_kwargs)
            if yerr is not None:
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color=line_kwargs.get("color", None),
                    ec="None",
                    alpha=fill_alpha,
                )
        else:
            line_kwargs.setdefault("marker", "o")
            if hollow:
                line_kwargs.setdefault("mfc", "None")
            line_kwargs.setdefault("capsize", 5)
            ax.errorbar(x, y, yerr, **line_kwargs)
        if sim_same_color:
            sim_kwargs["color"] = ax.get_lines()[-1].get_color()
            sim_kwargs["alpha"] = ax.get_lines()[-1].get_alpha()
            ax.plot(xsim, ysim, **sim_kwargs)
        legend_on = line_kwargs.get("label", None) is not None
    else:
        label = line_kwargs.pop("label", None)
        if label is None:
            label = [""] * n
        else:
            legend_on = True
            if isinstance(label, str):
                label = [label] * n
            for i in range(n):
                label[i] = str(label[i])

        if len(x.shape) == 1:
            x = np.array([x] * n)
        colors = line_kwargs.pop("color", None)
        if isinstance(colors, str) or colors is None:
            colors = [colors]
        colors = cycle(colors)

        if "ls" in line_kwargs:
            ls = line_kwargs.pop("ls")
            index_ls_map = {i: ls for i in range(n)}

        for i in range(n):
            color = next(colors)
            ls = index_ls_map.get(i, None)
            if yerr is None or not errorbar:
                ax.plot(
                    x[i],
                    y[i],
                    color=color,
                    label=label[i],
                    ls=ls,
                    **line_kwargs,
                )
                if yerr is not None:
                    ax.fill_between(
                        x[i],
                        y[i] - yerr[i],
                        y[i] + yerr[i],
                        color=color,
                        ec="None",
                        alpha=0.2,
                    )
            else:
                line_kwargs.setdefault("marker", "o")
                if hollow:
                    line_kwargs.setdefault("mfc", "None")
                line_kwargs.setdefault("capsize", 5)
                ax.errorbar(
                    x[i],
                    y[i],
                    yerr[i],
                    color=color,
                    label=label[i],
                    ls=ls,
                    **line_kwargs,
                )

    if legend_on:
        ax.legend(
            labelspacing=0.1,
            framealpha=0.4,
            prop={"size": legend_size},
            loc=legend_loc,
            ncol=legend_ncol,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        fontdict = {}
        if title_color:
            fontdict["color"] = title_color
        if title_size:
            fontdict["fontsize"] = title_size
        ax.set_title(title, fontdict=fontdict)
    if fig_title is not None:
        fig.suptitle(fig_title)
    if xlim == "auto":
        xmin, xmax = lim_vars["xmin"], lim_vars["xmax"]
        dx = xmax - xmin
        xlim = xmin - 0.01 * dx, xmax + 0.01 * dx
    if ylim == "auto":
        ymin, ymax = lim_vars["ymin"], lim_vars["ymax"]
        dy = ymax - ymin
        ylim = ymin - 0.02 * dy, ymax + 0.02 * dy
    if xtick_unit_pi:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks, list(map(lambda x: rf"{x:.2f}$\pi$", ticks / np.pi)))
    if xlim is not None and not np.isclose(xlim[0], xlim[1]):
        ax.set_xlim(*xlim)
    if ylim is not None and not np.isclose(ylim[0], ylim[1]):
        ax.set_ylim(*ylim)
    ax.grid(grid)
    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    return ax


def plot2d(
    data,
    cmap="Spectral_r",
    ax=None,
    figsize=[7, 5],
    xlim=[0, 1],
    ylim=[0, 1],
    xlabel=None,
    ylabel=None,
    title=None,
    plot_cbar=True,
    clabel=None,
    vmin=None,
    vmax=None,
    interp="none",
    fontsize=None,
    convert_lim=True,
    horizontal=False,
    constrained_layout=True,
    return_cbar=False,
    log=False,
    norm=None,
):
    if not hasattr(data, "mask"):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError()
    xlim = [np.min(xlim), np.max(xlim)]
    ylim = [np.min(ylim), np.max(ylim)]
    if convert_lim:
        Ny, Nx = data.shape
        dx = (xlim[1] - xlim[0]) / max(Nx - 1, 1)
        dy = (ylim[1] - ylim[0]) / max(Ny - 1, 1)
        xlim = xlim.copy()
        ylim = ylim.copy()
        xlim[0] -= dx / 2
        ylim[0] -= dy / 2
        xlim[1] += dx / 2
        ylim[1] += dy / 2

    if ax is None:
        ax = subplot(figsize=figsize)
    fig = ax.figure
    if constrained_layout:
        fig.set_constrained_layout(True)

    if vmin is None:
        vmin = vmin or np.nanmin(data)
    if vmax is None:
        vmax = vmax or np.nanmax(data)
    if vmin > data.min():
        print("data.min() < vmin, the plotting result can be misleading")
    if vmax < data.max():
        print("data.max() > vmax, the plotting result can be misleading")
    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin = None
        vmax = None
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=list(xlim) + list(ylim),
        aspect="auto",
        interpolation=interp,
        norm=norm,
    )

    ax.set_xlabel(xlabel, {"fontsize": fontsize})
    ax.set_ylabel(ylabel, {"fontsize": fontsize})
    ax.set_title(title, fontsize=fontsize)

    cbar = None
    if plot_cbar:
        if horizontal:
            orientation = "horizontal"
        else:
            orientation = "vertical"
        cbar = fig.colorbar(im, ax=ax, orientation=orientation)
        cbar.set_label(clabel)
    if return_cbar:
        return ax, cbar

    return ax

