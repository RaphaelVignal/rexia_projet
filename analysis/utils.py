from math import ceil
from typing import Optional, Iterable

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv

DATA_PATH = '../Data/RH_dataset.csv'


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dataframe() -> DataFrame:
    return read_csv(DATA_PATH, delimiter=';')


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode_series_for_plot(s: pd.Series) -> tuple:
    """
    Return numeric values for plotting a Series.

    - Numeric series: returned as-is (float), no tick remapping needed.
    - Categorical/text series: mapped to integer codes with a tick label list.

    Returns:
        y_values    – float Series ready for plotting
        tick_pos    – list[int] | None
        tick_labels – list[str] | None
        is_numeric  – bool
    """
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        return s_num.astype(float), None, None, True

    s_cat = s.astype("string")
    categories = pd.Index(s_cat.dropna().unique())
    mapping = {cat: i for i, cat in enumerate(categories)}
    y = s_cat.map(mapping).astype(float)
    return y, list(mapping.values()), list(mapping.keys()), False


def _parse_x_series(s: pd.Series) -> pd.Series:
    """
    Coerce an x-column to the most specific type available:
    numeric → datetime → string (last resort).
    """
    x_num = pd.to_numeric(s, errors="coerce")
    if x_num.notna().any():
        return x_num
    x_dt = pd.to_datetime(s, errors="coerce")
    if x_dt.notna().any():
        return x_dt
    return s.astype("string")


def _apply_yticks(ax: plt.Axes, tick_pos, tick_labels, max_labels: int = 20) -> None:
    """Apply categorical y-axis tick labels when count ≤ max_labels."""
    if tick_pos is not None and tick_labels is not None and len(tick_labels) <= max_labels:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)


def _prepare_df_for_plot(df: DataFrame, x_col: str, *y_cols: str) -> DataFrame:
    """
    Select [x_col, *y_cols], coerce x with _parse_x_series, drop NaN in x, sort by x.
    Supports one or more y columns.
    """
    cols = [x_col, *y_cols]
    tmp = df[cols].copy()
    tmp[x_col] = _parse_x_series(tmp[x_col])
    return tmp.dropna(subset=[x_col]).sort_values(x_col)


def _build_shared_y_encoding(
    prepared: list[DataFrame],
    y_col: str,
) -> tuple[list[DataFrame], list | None, list | None]:
    """
    Build a consistent y encoding across n DataFrames for the same column.

    If all values are numeric, assigns __y via direct numeric conversion.
    If categorical, builds a single global mapping so categories align across
    all series (i.e. the same label always maps to the same y position).

    Mutates each DataFrame in `prepared` in-place by adding a ``__y`` column.

    Returns:
        (prepared, y_ticks, y_labels)
    """
    y_all = pd.concat([p[y_col] for p in prepared], ignore_index=True)
    y_all_num = pd.to_numeric(y_all, errors="coerce")

    if y_all_num.notna().all():
        for p in prepared:
            p["__y"] = pd.to_numeric(p[y_col], errors="coerce")
        return prepared, None, None

    y_all_cat = y_all.astype("string")
    categories = pd.Index(y_all_cat.dropna().unique())
    mapping = {cat: i for i, cat in enumerate(categories)}
    for p in prepared:
        p["__y"] = p[y_col].astype("string").map(mapping).astype(float)
    return prepared, list(mapping.values()), list(mapping.keys())


def _draw_resignation_vlines(
    ax: plt.Axes,
    df: DataFrame,
    x_col: str,
    color,
    resignation_col: str = "label",
) -> None:
    """
    Draw a dashed vertical line for every row where resignation_col == 1.
    Silently skips if the column is absent in df.
    """
    if resignation_col not in df.columns:
        return
    for x_val in df.loc[df[resignation_col] == 1, x_col]:
        ax.axvline(x_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7)


# ---------------------------------------------------------------------------
# Public backward-compat alias
# ---------------------------------------------------------------------------

def to_plot_values(series: pd.Series):
    """
    Backward-compatible wrapper around _encode_series_for_plot.
    Returns (y_values, tick_positions, tick_labels) — without is_numeric flag.
    """
    y, tick_pos, tick_labels, _ = _encode_series_for_plot(series)
    return y, tick_pos, tick_labels


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_distribution(df: DataFrame, column: str) -> None:
    from pandas.api.types import is_numeric_dtype
    s = df[column].dropna()
    plt.figure(figsize=(8, 4))
    if is_numeric_dtype(s):
        plt.title(f'Distribution of {column}')
        plt.hist(s, bins=20)
        plt.ylabel('Frequency')
    else:
        plt.title(f'Distribution of {column} - Ascending')
        counts = s.astype(str).value_counts().sort_index()
        plt.bar(counts.index, counts.values)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()


def plot_column_vs_anciennete(
    ax: plt.Axes,
    dataframes: DataFrame | list[DataFrame],
    x_col: str,
    y_col: str,
    labels: Optional[list[str]] = None,
    kinds: Optional[list[str]] = None,
    show_resignations: bool = False,
) -> None:
    """
    Plot y_col against x_col on a provided Matplotlib axis.

    Supports one or multiple DataFrames superposed on the same axis.
    Each series is x-normalised independently (origin forced to 0).
    Colors cycle through tab10.

    Args:
        ax:                 target Matplotlib axis.
        dataframes:         one DataFrame or a list of DataFrames.
        x_col:              x-axis column (must be numeric-coercible).
        y_col:              y-axis column.
        labels:             legend label per series. Shown only when multiple
                            series are present or explicitly provided.
        kinds:              "line" or "scatter" per series. Auto-detected from
                            dtype when None.
        show_resignations:  if True, draw a dashed vline (same color as the
                            series) at every x where the "label" column == 1.
                            Silently skipped for series missing that column.
    """
    if isinstance(dataframes, DataFrame):
        dataframes = [dataframes]

    n = len(dataframes)
    show_legend = labels is not None or n > 1
    labels = labels or [f"Series {i}" for i in range(n)]
    kinds = kinds or [None] * n

    cmap = plt.get_cmap("tab10")

    raw_tmps = []
    for df in dataframes:
        extra = ["label"] if show_resignations and "label" in df.columns else []
        cols = [x_col, y_col, *extra]
        tmp = df[cols].copy()
        tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
        tmp = tmp.dropna(subset=[x_col]).sort_values(x_col)
        tmp[x_col] -= tmp[x_col].min()  # origin at 0, independent per series
        raw_tmps.append(tmp)

    raw_tmps, y_ticks, y_labels = _build_shared_y_encoding(raw_tmps, y_col)

    for i, (tmp, label, kind) in enumerate(zip(raw_tmps, labels, kinds)):
        tmp = tmp.dropna(subset=["__y"])

        if kind is None:
            s_num = pd.to_numeric(tmp[y_col], errors="coerce")
            kind = "line" if s_num.notna().all() else "scatter"

        color = cmap(i % 10)
        kwargs: dict = {"label": label} if show_legend else {}
        if n > 1:
            kwargs["color"] = color

        if kind == "scatter":
            ax.scatter(tmp[x_col], tmp["__y"], s=18, alpha=0.8, **kwargs)
        else:
            ax.plot(tmp[x_col], tmp["__y"], marker="o", linewidth=1.5, **kwargs)

        if show_resignations:
            _draw_resignation_vlines(ax, tmp, x_col, color=color)

    _apply_yticks(ax, y_ticks, y_labels, max_labels=15)
    ax.set_title(y_col)
    ax.set_xlabel(x_col)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize="small")


def multiplot_all_columns_by_anciennete(
    dataframes: DataFrame | list[DataFrame],
    x_col: str = "Ancienneté groupe (années)",
    exclude_cols: Optional[Iterable[str]] = None,
    labels: Optional[list[str]] = None,
    kinds: Optional[list[str]] = None,
    show_resignations: bool = False,
    ncols: int = 3,
    height_per_row: float = 4,
    width_per_col: float = 5.2,
) -> None:
    """
    Plot all columns (except excluded ones) against x_col in a subplot grid.

    Supports one or multiple DataFrames — each subplot shows all series
    superposed on the same axis via plot_column_vs_anciennete.

    Args:
        dataframes:        one DataFrame or a list of DataFrames.
        x_col:             shared x-axis column (must exist in every DataFrame).
        exclude_cols:      columns to omit from the grid (x_col always excluded).
                           When show_resignations=True, "label" is also excluded
                           automatically (it is encoded as vlines instead).
        labels:            legend label per DataFrame.
        kinds:             "line" or "scatter" per DataFrame (auto-detected when None).
        show_resignations: passed through to plot_column_vs_anciennete.
        ncols:             number of subplot columns.
        height_per_row / width_per_col: figure sizing.
    """
    if isinstance(dataframes, DataFrame):
        dataframes = [dataframes]

    for df in dataframes:
        if x_col not in df.columns:
            raise ValueError(f"Column '{x_col}' not found in one of the dataframes.")

    seen: dict[str, None] = {}
    for df in dataframes:
        for col in df.columns:
            seen[col] = None

    excluded = set(exclude_cols or []) | {x_col}
    if show_resignations:
        excluded.add("label")  # encoded as vlines, not as a subplot

    y_cols = [c for c in seen if c not in excluded]
    if not y_cols:
        raise ValueError("No columns left to plot.")

    nrows = ceil(len(y_cols) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width_per_col * ncols, height_per_row * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()
    for i, col in enumerate(y_cols):
        valid_idx    = [j for j, df in enumerate(dataframes) if col in df.columns]
        valid_dfs    = [dataframes[j] for j in valid_idx]
        valid_labels = [labels[j] for j in valid_idx] if labels else None
        valid_kinds  = [kinds[j]  for j in valid_idx] if kinds  else None

        plot_column_vs_anciennete(
            axes_flat[i],
            dataframes=valid_dfs,
            x_col=x_col,
            y_col=col,
            labels=valid_labels,
            kinds=valid_kinds,
            show_resignations=show_resignations,
        )

    for j in range(len(y_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Data vs {x_col}", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_two_columns_by_x(
    df: DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    y1_kind: str = "line",
    y2_kind: str = "scatter",
    figsize: tuple[float, float] = (10, 4.5),
):
    """
    Plot y1 and y2 against x on two y-axes (left / right).
    Supports numeric or categorical y columns.

    Returns:
        (fig, ax1, ax2)
    """
    needed = [x_col, y1_col, y2_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = _prepare_df_for_plot(df, x_col, y1_col, y2_col)

    y1_vals, y1_ticks, y1_labels, _ = _encode_series_for_plot(tmp[y1_col])
    y2_vals, y2_ticks, y2_labels, _ = _encode_series_for_plot(tmp[y2_col])
    tmp["__y1"] = y1_vals
    tmp["__y2"] = y2_vals
    tmp = tmp.dropna(subset=["__y1", "__y2"])

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    if y1_kind == "scatter":
        h1 = ax1.scatter(tmp[x_col], tmp["__y1"], color="tab:blue", alpha=0.85, label=y1_col)
    else:
        h1, = ax1.plot(tmp[x_col], tmp["__y1"], color="tab:blue", marker="o", linewidth=1.5, label=y1_col)

    if y2_kind == "line":
        h2, = ax2.plot(tmp[x_col], tmp["__y2"], color="tab:red", marker="o", linewidth=1.5, label=y2_col)
    else:
        h2 = ax2.scatter(tmp[x_col], tmp["__y2"], color="tab:red", alpha=0.8, label=y2_col)

    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y1_col, color="tab:blue")
    ax2.set_ylabel(y2_col, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.25)
    ax2.grid(visible=False)

    _apply_yticks(ax1, y1_ticks, y1_labels)
    _apply_yticks(ax2, y2_ticks, y2_labels)

    ax1.set_title(f"{y1_col} and {y2_col} by {x_col}")
    ax1.legend([h1, h2], [y1_col, y2_col], loc="best")

    fig.tight_layout()
    return fig, ax1, ax2


def plot_n_series_by_x(
    dataframes: list[DataFrame],
    x_col: str,
    y_col: str,
    labels: Optional[list[str]] = None,
    kinds: Optional[list[str]] = None,
    show_resignations: bool = False,
    figsize: tuple[float, float] = (10, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the same y column from n DataFrames on one shared axis.

    Categorical y values are encoded with a single global mapping so that
    category positions are consistent across all series.
    Colors cycle through tab10 (wraps around for n > 10).

    Args:
        dataframes:        one DataFrame per series.
        x_col:             shared x-axis column name (present in every DataFrame).
        y_col:             shared y-axis column name (present in every DataFrame).
        labels:            legend label per series. Defaults to "Series 0", "Series 1", …
        kinds:             "line" or "scatter" per series. Defaults to "line" for all.
        show_resignations: if True, draw a dashed vline (same color as the series)
                           at every x where the "label" column == 1. Silently
                           skipped for series missing that column.
        figsize:           passed directly to plt.subplots.

    Returns:
        (fig, ax)
    """
    n = len(dataframes)
    if n == 0:
        raise ValueError("At least one DataFrame is required.")

    labels = labels or [f"Series {i}" for i in range(n)]
    kinds = kinds or ["line"] * n

    if len(labels) != n or len(kinds) != n:
        raise ValueError("`dataframes`, `labels`, and `kinds` must all have the same length.")

    for label, df in zip(labels, dataframes):
        missing = [c for c in (x_col, y_col) if c not in df.columns]
        if missing:
            raise ValueError(f"'{label}': missing columns {missing}")

    # Include "label" column in preparation when needed so it survives x-sorting.
    def _prep(df: DataFrame) -> DataFrame:
        extra = ["label"] if show_resignations and "label" in df.columns else []
        return _prepare_df_for_plot(df, x_col, y_col, *extra)

    prepared = [_prep(df) for df in dataframes]
    prepared, y_ticks, y_labels = _build_shared_y_encoding(prepared, y_col)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize)

    for i, (p, label, kind) in enumerate(zip(prepared, labels, kinds)):
        p = p.dropna(subset=["__y"])
        color = cmap(i % 10)
        if kind == "scatter":
            ax.scatter(p[x_col], p["__y"], color=color, alpha=0.85, label=label)
        else:
            ax.plot(p[x_col], p["__y"], color=color, marker="o", linewidth=1.5, label=label)

        if show_resignations:
            _draw_resignation_vlines(ax, p, x_col, color=color)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")
    ax.grid(True, alpha=0.25)
    _apply_yticks(ax, y_ticks, y_labels)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax


def plot_same_column_two_dataframes_by_x(
    df1: DataFrame,
    df2: DataFrame,
    x_col: str,
    y_col: str,
    df1_label: str = "df1",
    df2_label: str = "df2",
    df1_kind: str = "line",
    df2_kind: str = "line",
    figsize: tuple[float, float] = (10, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare the same y column from two DataFrames on one shared axis.
    Thin wrapper around plot_n_series_by_x for backward compatibility.

    Returns:
        (fig, ax)
    """
    return plot_n_series_by_x(
        dataframes=[df1, df2],
        x_col=x_col,
        y_col=y_col,
        labels=[df1_label, df2_label],
        kinds=[df1_kind, df2_kind],
        figsize=figsize,
    )