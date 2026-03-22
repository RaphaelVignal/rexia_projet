"""
Internal plotting plumbing.
Not part of the public API — import from employee.py instead.
"""
from math import ceil

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _shared_y_encoding(
    frames: list[DataFrame],
    y_col: str,
) -> tuple[list[DataFrame], list | None, list | None]:
    """
    Assign a ``__y`` column to every DataFrame in `frames` using one consistent
    encoding so categorical labels map to the same integer across all series.
    Mutates frames in-place.

    Returns: (frames, tick_pos, tick_labels)
    """
    y_all = pd.concat([f[y_col] for f in frames], ignore_index=True)
    y_num = pd.to_numeric(y_all, errors="coerce")

    if y_num.notna().all():
        for f in frames:
            f["__y"] = pd.to_numeric(f[y_col], errors="coerce")
        return frames, None, None

    cats = pd.Index(y_all.astype("string").dropna().unique())
    mapping = {c: i for i, c in enumerate(cats)}
    for f in frames:
        f["__y"] = f[y_col].astype("string").map(mapping).astype(float)
    return frames, list(mapping.values()), list(mapping.keys())


def _apply_yticks(ax: plt.Axes, tick_pos, tick_labels, max_labels: int = 20) -> None:
    if tick_pos and tick_labels and len(tick_labels) <= max_labels:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)


# ---------------------------------------------------------------------------
# Core plot worker
# ---------------------------------------------------------------------------

def _plot_series_on_ax(
    ax: plt.Axes,
    frames: list[DataFrame],
    x_col: str,
    y_col: str,
    labels: list[str],
    show_resignations: bool = False,
    show_markers: bool = True,
    color_offset: int = 0,
) -> None:
    """
    Draw n series onto `ax`. X values are plotted as-is — no shifting.
    Kind (line vs scatter) is auto-detected per series. Colors cycle through tab10.

    `show_resignations` → dashed vline at every row where ``label`` == 1.
    `show_markers`      → draw point markers on line plots (disable for dense cohort data).
    """
    n = len(frames)
    cmap = plt.get_cmap("tab10")
    show_legend = n > 1 or any(l for l in labels)

    prepped = []
    for f in frames:
        extra = ["label"] if show_resignations and "label" in f.columns else []
        cols = list(dict.fromkeys([x_col, y_col, *extra]))
        tmp = f[cols].copy()
        tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
        tmp = tmp.dropna(subset=[x_col]).sort_values(x_col)
        prepped.append(tmp)

    prepped, y_ticks, y_labels = _shared_y_encoding(prepped, y_col)

    for i, (tmp, label) in enumerate(zip(prepped, labels)):
        tmp = tmp.dropna(subset=["__y"])
        color = cmap((i + color_offset) % 10)
        kw = {"color": color, "label": label} if show_legend else {"color": color}

        is_numeric = pd.to_numeric(tmp[y_col], errors="coerce").notna().all()
        if is_numeric:
            ax.plot(tmp[x_col], tmp["__y"], marker="o" if show_markers else "",
                    linewidth=1.5, **kw)
        else:
            ax.scatter(tmp[x_col], tmp["__y"], s=18, alpha=0.8, **kw)

        if show_resignations and "label" in tmp.columns:
            for x_val in tmp.loc[tmp["label"] == 1, x_col]:
                ax.axvline(x_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

    _apply_yticks(ax, y_ticks, y_labels, max_labels=15)
    ax.set_xlabel(x_col)
    ax.set_title(y_col)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize="small")


# ---------------------------------------------------------------------------
# Grid worker
# ---------------------------------------------------------------------------

def _multiplot_grid(
    frames: list[DataFrame],
    x_col: str,
    labels: list[str],
    exclude_cols: set[str],
    show_resignations: bool = False,
    ncols: int = 3,
    height_per_row: float = 4.0,
    width_per_col: float = 5.2,
) -> None:
    """
    One subplot per column (union across all frames, minus excluded).
    Each subplot calls _plot_series_on_ax with the matching frames.
    """
    seen: dict[str, None] = {}
    for f in frames:
        for c in f.columns:
            seen[c] = None

    y_cols = [c for c in seen if c not in exclude_cols]
    if not y_cols:
        raise ValueError("No columns left to plot.")

    nrows = ceil(len(y_cols) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(width_per_col * ncols, height_per_row * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, col in enumerate(y_cols):
        valid = [(f, l) for f, l in zip(frames, labels) if col in f.columns]
        _plot_series_on_ax(
            axes_flat[i],
            frames=[v[0] for v in valid],
            x_col=x_col,
            y_col=col,
            labels=[v[1] for v in valid],
            show_resignations=show_resignations,
        )

    for j in range(len(y_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Toutes les colonnes vs {x_col}", y=1.02)
    fig.tight_layout()
    plt.show()
