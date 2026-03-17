from math import ceil
from typing import Optional, Iterable

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv

DATA_PATH = '../Data/RH_dataset.csv'


def load_dataframe() -> DataFrame:
    return read_csv(DATA_PATH, delimiter=';')


def _encode_series_for_plot(s: pd.Series) -> tuple:
    """
    Return numeric values for plotting a Series.

    - Numeric series: returned as-is (float), no tick remapping needed.
    - Categorical/text series: mapped to integer codes with a tick label list.

    Returns:
        y_values   – float Series ready for plotting
        tick_pos   – list[int] | None
        tick_labels– list[str] | None
        is_numeric – bool
    """
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        return s_num.astype(float), None, None, True

    s_cat = s.astype("string")
    categories = pd.Index(s_cat.dropna().unique())
    mapping = {label: i for i, label in enumerate(categories)}
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
    df: DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    """Plot y_col against x_col on a provided Matplotlib axis."""
    # Ancienneté is always numeric — skip the full _parse_x_series cascade.
    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col]).sort_values(x_col)

    y_vals, y_ticks, y_labels, is_numeric = _encode_series_for_plot(tmp[y_col])
    tmp["__y"] = y_vals
    tmp = tmp.dropna(subset=["__y"])

    if is_numeric:
        ax.plot(tmp[x_col], tmp["__y"], marker="o", linewidth=1.5)
    else:
        ax.scatter(tmp[x_col], tmp["__y"], s=18, alpha=0.8)
        _apply_yticks(ax, y_ticks, y_labels, max_labels=15)

    ax.set_title(y_col)
    ax.set_xlabel(x_col)
    ax.grid(True, alpha=0.25)


def multiplot_all_columns_by_anciennete(
    df: DataFrame,
    x_col: str = "Ancienneté groupe (années)",
    exclude_cols: Optional[Iterable[str]] = None,
    ncols: int = 3,
    height_per_row: float = 4,
    width_per_col: float = 5.2,
) -> None:
    """Plot all columns (except excluded ones) against x_col in a grid."""
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in dataframe.")

    excluded = set(exclude_cols or []) | {x_col}
    y_cols = [c for c in df.columns if c not in excluded]
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
        plot_column_vs_anciennete(axes_flat[i], df, x_col=x_col, y_col=col)
    for j in range(len(y_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"All columns vs {x_col}", y=1.02)
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
):
    """
    Compare the same y column from two DataFrames on one shared axis.
    Supports numeric or categorical y values.

    Returns:
        (fig, ax)
    """
    for name, df in ((df1_label, df1), (df2_label, df2)):
        missing = [c for c in (x_col, y_col) if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

    p1 = _prepare_df_for_plot(df1, x_col, y_col)
    p2 = _prepare_df_for_plot(df2, x_col, y_col)

    # Build a shared y encoding so categories align between both dataframes.
    y_all = pd.concat([p1[y_col], p2[y_col]], ignore_index=True)
    y_all_num = pd.to_numeric(y_all, errors="coerce")

    if y_all_num.notna().all():
        p1["__y"] = pd.to_numeric(p1[y_col], errors="coerce")
        p2["__y"] = pd.to_numeric(p2[y_col], errors="coerce")
        y_ticks, y_labels = None, None
    else:
        y_all_cat = y_all.astype("string")
        categories = pd.Index(y_all_cat.dropna().unique())
        mapping = {label: i for i, label in enumerate(categories)}
        p1["__y"] = p1[y_col].astype("string").map(mapping).astype(float)
        p2["__y"] = p2[y_col].astype("string").map(mapping).astype(float)
        y_ticks = list(mapping.values())
        y_labels = list(mapping.keys())

    p1 = p1.dropna(subset=["__y"])
    p2 = p2.dropna(subset=["__y"])

    fig, ax = plt.subplots(figsize=figsize)

    if df1_kind == "scatter":
        ax.scatter(p1[x_col], p1["__y"], color="tab:blue", alpha=0.85, label=df1_label)
    else:
        ax.plot(p1[x_col], p1["__y"], color="tab:blue", marker="o", linewidth=1.5, label=df1_label)

    if df2_kind == "scatter":
        ax.scatter(p2[x_col], p2["__y"], color="tab:orange", alpha=0.85, label=df2_label)
    else:
        ax.plot(p2[x_col], p2["__y"], color="tab:orange", marker="o", linewidth=1.5, label=df2_label)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col}: {df1_label} vs {df2_label} by {x_col}")
    ax.grid(True, alpha=0.25)
    _apply_yticks(ax, y_ticks, y_labels)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax