from functools import wraps
import os

import matplotlib as mpl

import platform

USE_TEX = True
LW = 1.2
MS = 3.5
MEW = 0.7
CAPSIZE = 3
LEGEND_SIZE = 6
FONTSIZE = 7
DASHES = (3, 2.5)


def article_style(**rcparams):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            original_params = mpl.rcParams.copy()
            mpl.rcParams.update(
                {
                    "font.size": FONTSIZE,
                    "axes.titlesize": FONTSIZE,
                    "axes.labelsize": FONTSIZE,
                    "xtick.labelsize": FONTSIZE,
                    "ytick.labelsize": FONTSIZE,
                    "legend.fontsize": FONTSIZE,
                    "mathtext.fontset": "custom",
                }
            )
            if not platform.system().lower() == "linux":
                mpl.rcParams.update(
                    {
                        "font.family": "Arial",
                        "mathtext.rm": "Arial",
                        "mathtext.it": "Arial:italic",
                    }
                )
            actual_rcparams = {}
            for key in rcparams:
                actual_key = key.replace("_", ".")
                actual_rcparams[actual_key] = rcparams[key]
            mpl.rcParams.update(actual_rcparams)
            result = plot_func(*args, **kwargs)
            mpl.rcParams.update(original_params)
            return result

        return wrapper

    return decorator


def get_circuit_style(gate_colors=None, texts=None):
    gate_colors = gate_colors or {}
    texts = texts or {}

    gate_colors.setdefault("cz", "#3180bd")
    gate_colors.setdefault("_measure", "M")

    style = {"displaycolor": {}, "displaytext": {}}
    for gate_name, color in gate_colors.items():
        style["displaycolor"][gate_name] = (color, "#000000")
    for gate_name, text in texts.items():
        style["displaytext"][gate_name] = text

    return style


def format_legend(ax, size=LEGEND_SIZE, handles=None, labels=None, labelspacing=0.5, usetex=False, **kwargs):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if usetex:
        mpl.rcParams['text.usetex'] = True

    ax.legend(
        handles,
        labels,
        labelspacing=labelspacing,
        handletextpad=0.3,
        framealpha=0.2,
        prop={"size": size},
        **kwargs
    )

    mpl.rcParams['text.usetex'] = False