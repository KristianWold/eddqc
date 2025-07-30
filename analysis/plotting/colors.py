COLORS = {
    "default": [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "royalblue",
        "crimson",
        "seagreen",
        "gold",
        "orangered",
    ],
}

DC = COLORS["default"]


def color_palette(name=None, as_cmap=False):
    import seaborn as sns

    if name in COLORS:
        palette = COLORS[name]
    else:
        palette = name
    return sns.color_palette(palette, as_cmap=as_cmap)
