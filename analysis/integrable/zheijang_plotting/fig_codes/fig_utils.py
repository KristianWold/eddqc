from copy import deepcopy
import re

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


class AxesGroup:
    def __init__(
        self,
        num_axes=1,
        cols_or_rows=4,
        max_rows_or_cols_per_fig=5,
        figs=None,
        ax_width=None,
        ax_height=None,
        col_major=False,
        mark_index=True,
        footnote=None,
        indexes_3d=None,
        init=True,
    ) -> None:
        self.num_axes = num_axes
        self.cols_or_rows = cols_or_rows
        self.max_rows_or_cols_per_fig = max_rows_or_cols_per_fig
        self.ax_width = ax_width
        self.ax_height = ax_height
        self.col_major = col_major
        self.mark_index = mark_index
        self.footnote = footnote
        self.axes = None
        self.figs = figs
        self.inited = False
        if init:
            self.init_axes(indexes_3d=indexes_3d)

    def init(self, figs, axes):
        assert not self.inited
        if not np.iterable(figs):
            figs = [figs]
        if not np.iterable(axes):
            axes = [axes]
        self.inited = True
        self.figs = np.array(figs)
        self.axes = np.array(axes)
        return self

    def init_axes(self, indexes_3d=None):
        assert not self.inited
        self.inited = True
        if self.col_major:
            self.figs, self.axes = smart_axes_col_major(
                num_axes=self.num_axes,
                rows=self.cols_or_rows,
                max_cols_per_fig=self.max_rows_or_cols_per_fig,
                ax_width=self.ax_width,
                ax_height=self.ax_height,
                mark_index=self.mark_index,
                figs=self.figs,
                footnote=self.footnote,
                indexes_3d=indexes_3d,
            )
        else:
            self.figs, self.axes = smart_axes(
                num_axes=self.num_axes,
                cols=self.cols_or_rows,
                max_rows_per_fig=self.max_rows_or_cols_per_fig,
                ax_width=self.ax_width,
                ax_height=self.ax_height,
                mark_index=self.mark_index,
                figs=self.figs,
                footnote=self.footnote,
                indexes_3d=indexes_3d,
            )
        return self.figs, self.axes

    def broadcast(self, ax_func_name, condition=None, *args, **kwargs):
        for ax in self.axes:
            if condition is None or condition(ax):
                getattr(ax, ax_func_name)(*args, **kwargs)

    def tick_params(
        self, axis="both", direction="out", bottom=True, left=True, *args, **kwargs
    ):
        self.broadcast(
            "tick_params",
            axis=axis,
            direction=direction,
            bottom=bottom,
            left=left,
            *args,
            **kwargs,
        )

    def set_xticks(
        self,
        ticks=None,
        labels=None,
        xlim=None,
        xlim_pad=None,
        xlim_pad_ratio=None,
        sharex=True,
        fontsize=7,
        axes=None,
    ):
        if xlim is not None:
            xlim = [np.min(xlim), np.max(xlim)]
        axes = self.convert_axes(axes)
        if xlim is not None and xlim_pad is None:
            if xlim_pad_ratio is not None:
                xlim_pad = (np.max(xlim) - np.min(xlim)) * xlim_pad_ratio
            else:
                xlim_pad = 0
        for ax in axes:
            if ticks is not None:
                if labels is not None and (not sharex or self.is_bottom(ax, axes=axes)):
                    ax.set_xticks(ticks, labels=labels, fontsize=fontsize)
                else:
                    ax.set_xticks(ticks)
            if sharex and not self.is_bottom(ax, axes=axes):
                ax.tick_params(labelbottom=False)
            if xlim is not None:
                _xlim = deepcopy(xlim)
                _xlim[0] -= xlim_pad
                _xlim[1] += xlim_pad
                ax.set_xlim(_xlim)

    def set_yticks(
        self,
        ticks=None,
        labels=None,
        ylim=None,
        ylim_pad=None,
        ylim_pad_ratio=None,
        sharey=True,
        fontsize=7,
        axes=None,
        **kwargs,
    ):
        if ylim is not None:
            ylim = [np.min(ylim), np.max(ylim)]
        axes = self.convert_axes(axes)
        if ylim is not None and ylim_pad is None:
            if ylim_pad_ratio is not None:
                ylim_pad = (np.max(ylim) - np.min(ylim)) * ylim_pad_ratio
            else:
                ylim_pad = 0
        for ax in axes:
            if ticks is not None:
                if labels is not None and (not sharey or self.is_left(ax, axes=axes)):
                    ax.set_yticks(ticks, labels=labels, fontsize=fontsize, **kwargs)
                else:
                    ax.set_yticks(ticks, **kwargs)
            if sharey and not self.is_left(ax, axes=axes):
                ax.tick_params(labelleft=False)
            if ylim is not None:
                _ylim = deepcopy(ylim)
                _ylim[0] -= ylim_pad
                _ylim[1] += ylim_pad
                ax.set_ylim(_ylim)

    def set_xlabel(
        self, xlabel, sharex=True, fontsize=None, clean_others=True, axes=None, **kwargs
    ):
        axes = self.convert_axes(axes)
        if fontsize:
            kwargs.setdefault("fontdict", {})["fontsize"] = fontsize
        self.broadcast(
            "set_xlabel",
            lambda ax: ax in axes and (not sharex or self.is_bottom(ax, axes=axes)),
            xlabel,
            **kwargs,
        )
        if sharex and clean_others:
            self.broadcast(
                "set_xlabel",
                lambda ax: ax in axes and not self.is_bottom(ax, axes=axes),
                None,
            )

    def set_ylabel(
        self,
        ylabel,
        sharey=True,
        fontsize=None,
        clean_others=False,
        axes=None,
        **kwargs,
    ):
        axes = self.convert_axes(axes)
        if fontsize:
            kwargs.setdefault("fontdict", {})["fontsize"] = fontsize
        self.broadcast(
            "set_ylabel",
            lambda ax: ax in axes and (not sharey or self.is_left(ax, axes=axes)),
            ylabel,
            **kwargs,
        )
        if sharey and clean_others:
            self.broadcast(
                "set_ylabel",
                lambda ax: ax in axes and not self.is_left(ax, axes=axes),
                None,
            )

    def remove_legend(self, axes=None):
        axes = self.convert_axes(axes)
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    def convert_axes(self, axes):
        if axes is None:
            axes = self.axes
        elif isinstance(axes, str):
            if result := re.fullmatch("col[_]?(.*)", axes):
                col = int(result[1])
                axes = [ax for ax in self.axes if self.col(ax) == col]
            elif result := re.fullmatch("row[_]?(.*)", axes):
                row = int(result[1])
                axes = [ax for ax in self.axes if self.row(ax) == row]
            else:
                raise
        else:
            if not np.iterable(axes):
                axes = [axes]
            new_axes = []
            for i in range(len(axes)):
                if isinstance(axes[i], str):
                    new_axes.extend(self.convert_axes(axes[i]))
                elif np.isscalar(axes[i]):
                    new_axes.append(self.axes[axes[i]])
                else:
                    # assert axes[i] in self.axes
                    new_axes.append(axes[i])
            axes = new_axes
        return axes

    def annot_alphabet(self, **kwargs):
        annot_alphabet(self.axes, **kwargs)

    def grid(self, *args, **kwargs):
        self.broadcast("grid", None, *args, **kwargs)

    def clear(self):
        self.broadcast("clear")

    @property
    def num_figs(self):
        return len(self.figs)

    def is_bottom(self, ax, axes=None):
        """
        a b c
        d e

        Then c, d and e are at the bottom
        """
        if self.col_major:
            raise NotImplementedError()
        if axes is None:
            axes = self.axes
        axes = self.convert_axes(axes)
        ax = self.convert_axes(ax)[0]
        fig_indexes = [self.get_fig_index(i) for i in axes]
        fig_index = self.get_fig_index(ax)
        axes = [
            axes[i] for i in range(len(axes)) if fig_indexes[i] == fig_index
        ]  # make sure comparing on the same fig
        assert ax in axes
        index = list(axes).index(ax)
        num_axes = len(axes)

        is_bottom = True
        rows = [self.row(i) for i in axes]
        cols = [self.col(i) for i in axes]
        row = rows[index]
        col = cols[index]
        for i in range(num_axes):
            if i == index:
                continue
            if rows[i] > row and cols[i] == col:
                is_bottom = False
        return is_bottom

    def is_left(self, ax, axes=None):
        """
        a b c
        d e

        Then a and d are at the left
        """
        if self.col_major:
            raise NotImplementedError()
        assert ax in self.axes
        index = list(self.axes).index(ax)
        return index % self.cols_or_rows == 0

    def is_right(self, ax, axes=None):
        """
        a b c
        d e

        Then c and e are at the right
        """
        if self.col_major:
            raise NotImplementedError()
        assert ax in self.axes
        index = list(self.axes).index(ax)
        return (index + 1) % self.cols_or_rows == 0 or index == self.num_axes - 1

    def row(self, ax):
        assert ax in self.axes
        assert not self.col_major
        index = list(self.axes).index(ax)
        index = index % (self.max_rows_or_cols_per_fig * self.cols_or_rows)
        return index // self.cols_or_rows

    def col(self, ax):
        assert ax in self.axes
        assert not self.col_major
        index = list(self.axes).index(ax)
        return index % self.cols_or_rows

    def get_fig_index(self, ax):
        i = list(self.axes).index(ax)
        num_axes_per_fig = self.max_rows_or_cols_per_fig * self.cols_or_rows
        return i // num_axes_per_fig

    def tight_layout(self, pad=0):
        tight_layout_all(self.figs, pad=pad)

    def set_constrained_layout(self):
        constrained_layout_all(self.figs)

    def move_to_monitor(self, monitor_index=0, full_screen=False):
        move_to_monitor(self.figs, monitor_index=monitor_index, full_screen=full_screen)


def ceildiv(n, d):
    return -(n // -d)


def write_footnote(fig, text, fontsize=8, upper=False):
    text = str(text)
    child = fig.get_children()[-1]
    if isinstance(child, mpl.text.Text) and child.get_text() == text:
        return
    if upper:
        fig.text(
            1,
            1,
            text,
            transform=fig.transFigure,
            ha="right",
            va="top",
            fontsize=fontsize,
        )

    else:
        fig.text(
            1,
            0.004,
            text,
            transform=fig.transFigure,
            ha="right",
            va="bottom",
            fontsize=fontsize,
        )


def smart_axes(
    num_axes=1,
    cols=6,
    max_rows_per_fig=4,
    ax_width=None,
    ax_height=None,
    axis_width=None,  # for compatibility
    axis_height=None,  # for compatibility
    mark_index=True,
    figs=None,
    footnote=None,
    indexes_3d=None,
):
    if num_axes <= 0:
        return
    if ax_width is not None:
        axis_width = ax_width
    if ax_height is not None:
        axis_height = ax_height

    max_cols_per_fig = cols
    if num_axes == 1:
        rows = cols = 1
        num_figs = 1
    else:
        max_axes_per_fig = cols * max_rows_per_fig
        num_figs = ceildiv(num_axes, max_axes_per_fig)
        if num_figs == 1:
            rows = ceildiv(num_axes, cols)
        else:
            rows = ceildiv(num_axes % max_axes_per_fig, cols)
            if rows == 0:
                rows = max_rows_per_fig

    if num_axes <= max_cols_per_fig // 2:
        if axis_width is None:
            axis_width = 6
        ratio = 0.6
    else:
        if axis_width is None:
            axis_width = 4.5
        ratio = 0.7
    if axis_height is None:
        axis_height = axis_width * ratio

    if figs is None:
        figs = []
        create_figs = True
    else:
        if not np.iterable(figs):
            figs = [figs]
        assert len(figs) >= num_figs
        create_figs = False
    axes = []
    curr_fig = 0
    for _ in range(num_figs - 1):
        if create_figs:
            fig = plt.figure(
                figsize=(axis_width * cols, axis_height * max_rows_per_fig)
            )
            figs.append(fig)
        else:
            fig = figs[curr_fig]
            curr_fig += 1
        for i in range(max_axes_per_fig):
            if indexes_3d is not None and len(axes) in indexes_3d:
                projection = "3d"
            else:
                projection = None
            axes.append(
                fig.add_subplot(max_rows_per_fig, cols, i + 1, projection=projection)
            )

    num_remaining_axes = num_axes - len(axes)
    cols_in_last_figure = min(num_remaining_axes, cols)
    if create_figs:
        fig = plt.figure(figsize=(axis_width * cols_in_last_figure, axis_height * rows))
        figs.append(fig)
    else:
        fig = figs[curr_fig]
    for i in range(num_remaining_axes):
        if indexes_3d is not None and len(axes) in indexes_3d:
            projection = "3d"
        else:
            projection = None
        axes.append(
            fig.add_subplot(rows, cols_in_last_figure, i + 1, projection=projection)
        )

    if mark_index and len(figs) > 1:
        for i, fig in enumerate(figs):
            fig.text(
                1,
                1,
                str(i),
                transform=fig.transFigure,
                ha="right",
                va="top",
                fontsize=9,
            )
    if footnote is not None:
        for fig in figs:
            write_footnote(fig, footnote)

    return figs, np.array(axes)


def smart_axes_col_major(
    num_axes=1,
    rows=4,
    max_cols_per_fig=5,
    ax_width=None,
    ax_height=None,
    axis_width=None,  # for compatibility
    axis_height=None,  # for compatibility
    mark_index=True,
    footnote=None,
    figs=None,
    indexes_3d=None,
):
    if num_axes <= 0:
        return
    if ax_width is not None:
        axis_width = ax_width
    if ax_height is not None:
        axis_height = ax_height
    if num_axes == 1:
        rows = cols = 1
        num_figs = 1
    else:
        max_axes_per_fig = rows * max_cols_per_fig
        num_figs = ceildiv(num_axes, max_axes_per_fig)
        if num_figs == 1:
            cols = ceildiv(num_axes, rows)
        else:
            cols = ceildiv(num_axes % max_axes_per_fig, rows)
            if cols == 0:
                cols = max_cols_per_fig

    if axis_height is None:
        if num_axes <= rows // 2:
            axis_height = 3
        else:
            axis_height = 2.3
    if axis_width is None:
        axis_width = axis_height / 0.9

    axes = []
    if figs is None:
        figs = []
        create_figs = True
    else:
        if not np.iterable(figs):
            figs = [figs]
        assert len(figs) >= num_figs
        create_figs = False
    curr_fig = 0
    for _ in range(num_figs - 1):
        if create_figs:
            fig = plt.figure(
                figsize=(axis_width * max_cols_per_fig, axis_height * rows)
            )
            figs.append(fig)
        else:
            fig = figs[curr_fig]
            curr_fig += 1
        for i in range(max_axes_per_fig):
            if indexes_3d is not None and len(axes) in indexes_3d:
                projection = "3d"
            else:
                projection = None
            axes.append(
                fig.add_subplot(
                    rows,
                    max_cols_per_fig,
                    _col_major_index_to_row_major_index(i, rows, max_cols_per_fig) + 1,
                    projection=projection,
                )
            )

    num_remaining_axes = num_axes - len(axes)
    rows_in_last_figure = min(num_remaining_axes, rows)
    if create_figs:
        fig = plt.figure(figsize=(axis_width * cols, axis_height * rows_in_last_figure))
        figs.append(fig)
    else:
        fig = figs[curr_fig]
    for i in range(num_axes - len(axes)):
        if indexes_3d is not None and len(axes) in indexes_3d:
            projection = "3d"
        else:
            projection = None
        axes.append(
            fig.add_subplot(
                rows_in_last_figure,
                cols,
                _col_major_index_to_row_major_index(i, rows_in_last_figure, cols) + 1,
                projection=projection,
            )
        )
    if mark_index and len(figs) > 1:
        for i, fig in enumerate(figs):
            fig.text(
                1,
                1,
                str(i),
                transform=fig.transFigure,
                ha="right",
                va="top",
                fontsize=9,
            )
    if footnote is not None:
        for fig in figs:
            write_footnote(fig, footnote)
    return figs, axes


def _col_major_index_to_row_major_index(index, rows, cols):
    i, j = np.unravel_index(index, (rows, cols), order="F")
    return i * cols + j


def tight_layout_all(figs, pad=0):
    if not np.iterable(figs):
        figs = [figs]
    for fig in figs:
        fig.tight_layout(pad=pad)


def constrained_layout_all(figs):
    if not np.iterable(figs):
        figs = [figs]
    for fig in figs:
        fig.set_constrained_layout(1)


def subplot(figsize=None, footnote=None, is_3d=False):
    fig = plt.figure(figsize=figsize)
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    if footnote is not None:
        write_footnote(fig, footnote)
    return ax


def move_to_monitor(fig=None, monitor_index=0, full_screen=False):
    try:
        import screeninfo

        if fig is None:
            fm = plt.get_current_fig_manager()
        elif np.iterable(fig):
            for f in fig:
                move_to_monitor(
                    fig=f, monitor_index=monitor_index, full_screen=full_screen
                )
            return
        else:
            fm = fig.canvas.manager
        monitor = screeninfo.get_monitors()[monitor_index]
        fm.window.move(monitor.x, monitor.y)
        if full_screen:
            fm.window.showMaximized()
    except:
        pass


def fig_in_a4(width, height, dpi=200):
    size = get_a4_size()
    if width == "2col":
        width = 1
    elif width == "1col":
        width = 8.8 / 18
    else:
        assert isinstance(width, (int, float))

    return plt.figure(figsize=(size[0] * width, size[1] * height), dpi=dpi)


def get_a4_size():
    """
    The size of A4 paper is (21cm, 29.7cm), but we need to account for the margin.
    nature standard: 2 column: width=18cm, 1 column: width=8.8cm
    """
    # width = 21.0
    width = 18
    height = 29.7
    cm = 1 / 2.54
    return (width * cm, height * cm)


def get_ax_size(ax, fraction=True):
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    if not fraction:
        return width, height
    fig_width, fig_height = get_fig_size(fig)
    return width / fig_width, height / fig_height


def get_fig_size(fig):
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width, bbox.height


def offset_ticklabels(ax, which="both", offset=0, ratio=1):
    # check_in(which, ["x", "y", "both"])
    if which == "both":
        keys = ["x", "y"]
    else:
        keys = [which]
    for which in keys:
        ticks = getattr(ax, f"get_{which}ticks")()
        labels = getattr(ax, f"get_{which}ticklabels")()
        labels = ax.get_xticklabels()
        getattr(ax, f"set_{which}ticks")(ratio * ticks + offset, labels)


def set_label_property(ax, which="both", pad=0, visible=None):
    # check_in(which, ["x", "y", "both"])
    if which == "both":
        keys = ["x", "y"]
    else:
        keys = [which]
    for which in keys:
        getattr(ax, f"set_{which}label")(
            getattr(ax, f"get_{which}label")(), labelpad=pad
        )
        if visible is not None:
            plt.setp(getattr(ax, f"get_{which}ticklabels")(), visible=visible)


def num_to_alphabet(num, upper=False):
    """
    start from 0
    """
    if upper:
        return chr(num + 65)
    return chr(num + 97)


def alphabet_to_num(alphabet):
    """
    start from 0
    """
    alphabet = alphabet.lower()
    return ord(alphabet) - 97


def annot_alphabet(
    axes=None,
    dx=0,
    dy=0,
    upper=False,
    transform="ax",
    fontname="Arial",
    weight="bold",
    share_top=None,
    left_bond_dict=None,
    top_dict=None,
    top_bond_dict=None,
    offset=0,
    dx_dict=None,
    dy_dict=None,
    **kwargs,
):
    """
    Use transform='fig' can reverse the position of axes. But it may not work properly
    when tight_layout=True.
    """
    assert np.iterable(axes)
    # check_in(transform, ["ax", "fig"])
    dx_dict = dx_dict or {}
    dy_dict = dy_dict or {}
    top_bond_dict = top_bond_dict or {}
    left_bond_dict = left_bond_dict or {}
    for d in [dx_dict, dy_dict, top_bond_dict, left_bond_dict]:
        for k in list(d):
            if isinstance(k, str):
                v = d.pop(k)
                k = alphabet_to_num(k) - offset
                d[k] = v
    for d in [top_bond_dict, left_bond_dict]:
        for k in list(d):
            if isinstance(d[k], str):
                d[k] = alphabet_to_num(d[k]) - offset
    if transform == "ax":
        for i, ax in enumerate(axes):
            ax.text(
                -0.1 + dx + dx_dict.get(i, 0),
                1.08 + dy + dy_dict.get(i, 0),
                num_to_alphabet(i + offset, upper=upper),
                fontname=fontname,
                weight=weight,
                transform=ax.transAxes,
                **kwargs,
            )
    else:
        if np.iterable(share_top) and not np.iterable(share_top[0]):
            share_top = [share_top]
        fig = axes[0].figure
        for i, ax in enumerate(axes):
            box = ax.get_position()
            left = box.xmin
            top = box.ymax
            if top_dict and i in top_dict:
                top = top_dict[i]
            elif top_bond_dict and i in top_bond_dict:
                top = axes[top_bond_dict[i]].get_position().ymax
            elif share_top:
                for st in share_top:
                    if i in st:
                        tops = [axes[a].get_position().ymax for a in st]
                        top = np.max(tops)
                        break
            if left_bond_dict and i in left_bond_dict:
                left = axes[left_bond_dict[i]].get_position().xmin + dx_dict.get(
                    left_bond_dict[i], 0
                )

            fig.text(
                left + dx + dx_dict.get(i, 0),
                top + dy + dy_dict.get(i, 0),
                num_to_alphabet(i + offset, upper=upper),
                fontname=fontname,
                weight=weight,
                transform=fig.transFigure,
                **kwargs,
            )


def add_fix_ax(fig, rect, loc="lower left", is_3d=False):
    from mpl_toolkits.axes_grid1.inset_locator import AnchoredSizeLocator

    if is_3d:
        ax = fig.add_axes(rect, projection="3d")
    else:
        ax = fig.add_axes(rect)
    axes_locator = AnchoredSizeLocator(
        rect,
        "100%",
        "100%",
        loc=loc,
        bbox_transform=fig.transFigure,
        borderpad=0,
    )

    ax.set_axes_locator(axes_locator)
    return ax


def add_ax(fig, left, bottom, width, height, is_3d=False):
    if is_3d:
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    else:
        ax = fig.add_axes([0, 0, 1, 1])
    pos = ax.get_position()
    ax.set_position([pos.xmin + left, pos.ymin + bottom, width, height])
    return ax
