from math import ceil
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from utils import _plot_series_on_ax, _multiplot_grid

COL_MATRICULE          = 'matricule'
COL_SALARY_EUROS       = 'Salaire (Euros)'
COL_VEHICLE            = 'Véhicule'
COL_HIERARCHY_LEVEL    = 'Niveau hiérarchique'
COL_AGE_YEARS          = 'Âge (années)'
COL_ESTABLISHMENT      = 'Etablissement'
COL_SENIORITY_GROUP_YEARS  = 'Ancienneté groupe (années)'
COL_CONTRACT_START_YEARS   = 'Début de contrat (années)'
COL_LAST_RAISE_MONTHS      = 'Dernière augmentation (mois)'
COL_LAST_PROMOTION_MONTHS  = 'Dernière promotion (mois)'
COL_JOB_FAMILY         = "Famille d'emploi"
COL_HAS_RESIGNED       = "label"
COL_CUSTOM_TIME        = "Temps (années)"

COLUMN_DTYPES = {
    COL_MATRICULE:             int,
    COL_HIERARCHY_LEVEL:       int,
    COL_AGE_YEARS:             int,
    COL_VEHICLE:               int,
    COL_ESTABLISHMENT:         str,
    COL_JOB_FAMILY:            str,
    COL_SALARY_EUROS:          float,
    COL_SENIORITY_GROUP_YEARS: float,
    COL_CONTRACT_START_YEARS:  float,
    COL_LAST_RAISE_MONTHS:     float,
    COL_LAST_PROMOTION_MONTHS: float,
}

# Columns that are intentionally categorical despite having numeric-looking values.
CATEGORICAL_COLS = {COL_VEHICLE, COL_ESTABLISHMENT, COL_JOB_FAMILY}

# Columns always hidden from multiplot grids.
_MULTIPLOT_EXCLUDE = {COL_MATRICULE, COL_HAS_RESIGNED}

def fix_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors='raise')
    return df


def _add_custom_time(df: pd.DataFrame) -> None:
    """
    Add COL_CUSTOM_TIME in-place: years elapsed since the employee's first
    observation, derived from COL_SENIORITY_GROUP_YEARS.
    Works on a single-employee or multi-employee DataFrame (via groupby).
    No-op if the column already exists.
    """
    if COL_CUSTOM_TIME in df.columns:
        return

    def _compute(grp: pd.DataFrame) -> pd.Series:
        seniority = grp[COL_SENIORITY_GROUP_YEARS]
        return seniority - seniority.min()

    if COL_MATRICULE in df.columns and df[COL_MATRICULE].nunique() > 1:
        df[COL_CUSTOM_TIME] = (
            df.groupby(COL_MATRICULE, group_keys=False).apply(_compute)
        )
    else:
        df[COL_CUSTOM_TIME] = _compute(df)


def _is_numeric_col(column: str, df: pd.DataFrame) -> bool:
    """
    Authoritative numeric check: consults COLUMN_DTYPES first,
    falls back to dtype inference for undeclared columns.
    Columns in CATEGORICAL_COLS are always treated as categorical.
    """
    if column in CATEGORICAL_COLS:
        return False
    declared = COLUMN_DTYPES.get(column)
    if declared is not None:
        return declared in (int, float)
    return is_numeric_dtype(df[column].dropna())


# ---------------------------------------------------------------------------
# Employee
# ---------------------------------------------------------------------------

class Employee:
    """
    Single-employee view into the HR dataset.

    Public API
    ----------
    emp.plot(column, x)
    emp.plot_2(column1, column2, x)
    emp.compare(others, column, x, show_resignations)
    emp.multiplot(others, x, exclude_cols, show_resignations)
    Employee.compare_cohorts(df, column, x_col, n_bins)
    Employee.cohort_multiplot(df, order_by, n_bins, exclude_cols)
    """

    def __init__(self, df: pd.DataFrame, matricule: int, name: str | None = None):
        self.df = df[df[COL_MATRICULE] == matricule].copy()
        fix_dataframe_types(self.df)
        self.matricule = matricule
        self.name = name
        self.has_resigned = bool(self.df[COL_HAS_RESIGNED].sum() > 0)
        self.df.sort_values(COL_SENIORITY_GROUP_YEARS, inplace=True)
        _add_custom_time(self.df)

    @staticmethod
    def get_random(
        df: pd.DataFrame,
        has_resigned: bool | None = None,
        name: str | None = None,
    ) -> 'Employee':
        """Return a randomly sampled Employee, optionally filtered by resignation status."""
        pool = (
            df[COL_MATRICULE] if has_resigned is None
            else df.loc[df[COL_HAS_RESIGNED] == int(has_resigned), COL_MATRICULE]
        )
        if pool.empty:
            raise ValueError("No employee matches the requested resignation filter.")
        return Employee(df, pool.sample(n=1).iloc[0], name=name)

    def _label(self) -> str:
        return self.name or f"Matricule {self.matricule}"

    @staticmethod
    def _normalise_others(
        others: 'Employee | list[Employee] | None',
    ) -> 'list[Employee]':
        if others is None:
            return []
        return [others] if isinstance(others, Employee) else list(others)

    def plot(
        self,
        column: str,
        x: str = COL_CUSTOM_TIME,
        show_resignations: bool = False,
    ) -> None:
        """
        Plot one column over time for this employee alone.

        Usage:
            emp.plot(COL_SALARY_EUROS)
            emp.plot(COL_HIERARCHY_LEVEL, x=COL_CUSTOM_TIME, show_resignations=True)
        """
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _plot_series_on_ax(
            ax, [self.df], x_col=x, y_col=column,
            labels=[self._label()], show_resignations=show_resignations,
        )
        fig.tight_layout()
        fig.show()

    def plot_2(
        self,
        column1: str,
        column2: str,
        x: str = COL_CUSTOM_TIME,
        show_resignations: bool = False,
    ) -> None:
        """
        Plot two columns on dual Y axes for this employee.

        column1 → left axis (blue), column2 → right axis (orange).

        Usage:
            emp.plot_2(COL_SALARY_EUROS, COL_HIERARCHY_LEVEL)
            emp.plot_2(COL_SALARY_EUROS, COL_AGE_YEARS, x=COL_CUSTOM_TIME)
        """
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax2 = ax.twinx()

        _plot_series_on_ax(
            ax, [self.df], x_col=x, y_col=column1,
            labels=[""], show_resignations=show_resignations, color_offset=0,
        )
        _plot_series_on_ax(
            ax2, [self.df], x_col=x, y_col=column2,
            labels=[""], show_resignations=False, color_offset=1,
        )

        ax2.set_title("")
        ax2.grid(False)
        ax.set_title(f"{column1}  /  {column2}")

        cmap = plt.get_cmap("tab10")
        handles = [
            plt.Line2D([0], [0], color=cmap(0), linewidth=1.5, marker="o", label=column1),
            plt.Line2D([0], [0], color=cmap(1), linewidth=1.5, marker="o", label=column2),
        ]
        ax.legend(handles=handles, loc="best", fontsize="small")

        fig.tight_layout()
        fig.show()

    def compare(
        self,
        others: 'Employee | list[Employee]',
        column: str,
        x: str = COL_CUSTOM_TIME,
        show_resignations: bool = False,
    ) -> None:
        """
        Superpose one column from self and one or more other employees.

        Usage:
            emp1.compare(emp2, COL_SALARY_EUROS)
            emp1.compare([emp2, emp3], COL_SALARY_EUROS, x=COL_CUSTOM_TIME)
            emp1.compare(emp2, COL_SALARY_EUROS, show_resignations=True)
        """
        all_emps = [self, *Employee._normalise_others(others)]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _plot_series_on_ax(
            ax,
            frames=[e.df for e in all_emps],
            x_col=x,
            y_col=column,
            labels=[e._label() for e in all_emps],
            show_resignations=show_resignations,
        )
        fig.tight_layout()
        fig.show()

    def multiplot(
        self,
        others: 'Employee | list[Employee] | None' = None,
        x: str = COL_CUSTOM_TIME,
        exclude_cols: Optional[Iterable[str]] = None,
        show_resignations: bool = False,
    ) -> None:
        """
        Grid of subplots — every column vs x.
        COL_MATRICULE and COL_HAS_RESIGNED are always excluded.

        Usage:
            emp.multiplot()
            emp.multiplot(emp2, show_resignations=True)
            emp.multiplot([emp2, emp3], x=COL_CUSTOM_TIME)
            emp.multiplot(exclude_cols={COL_VEHICLE})
        """
        all_emps = [self, *Employee._normalise_others(others)]
        excluded = _MULTIPLOT_EXCLUDE | set(exclude_cols or [])
        labels = [e._label() for e in all_emps] if len(all_emps) > 1 else [""]
        _multiplot_grid(
            frames=[e.df for e in all_emps],
            x_col=x,
            labels=labels,
            exclude_cols=excluded,
            show_resignations=show_resignations,
        )

    @staticmethod
    def cohort_average(
        df: pd.DataFrame,
        has_resigned: bool,
        x_col: str = COL_CUSTOM_TIME,
        n_bins: int = 50,
    ) -> pd.DataFrame:
        """
        Average career trajectory for a cohort on a true time axis.

        Each employee is interpolated onto a shared grid from 0 to the longest
        observed career in the cohort. Values outside an employee's observed
        range are left as NaN, so the average at each time point only reflects
        employees who were actually present then. Non-numeric columns are dropped.

        Returns a DataFrame with:
            COL_CUSTOM_TIME – time grid in years
            "<col>"         – mean value per bin (NaN where no employees remain)
        """
        cohort = df[df[COL_HAS_RESIGNED] == int(has_resigned)].copy()
        fix_dataframe_types(cohort)
        _add_custom_time(cohort)

        matricules = cohort[COL_MATRICULE].unique()
        if len(matricules) == 0:
            raise ValueError(f"No employees found with has_resigned={has_resigned}.")

        exclude = {COL_MATRICULE, COL_HAS_RESIGNED, x_col}
        numeric_cols = [
            c for c in cohort.columns
            if c not in exclude and _is_numeric_col(c, cohort)
        ]

        work = (
            cohort[[COL_MATRICULE, x_col, *numeric_cols]]
            .dropna(subset=[x_col])
            .sort_values([COL_MATRICULE, x_col])
        )

        grid = np.linspace(0, work[x_col].max(), n_bins)
        n_emp = len(matricules)
        mat_index = {m: i for i, m in enumerate(matricules)}
        buffers = {c: np.full((n_emp, n_bins), np.nan) for c in numeric_cols}

        for mat, grp in work.groupby(COL_MATRICULE, sort=False):
            x_raw = grp[x_col].to_numpy(dtype=float)
            if len(grp) < 2 or x_raw.max() == x_raw.min():
                continue
            row = mat_index[mat]
            for col in numeric_cols:
                vals = np.interp(grid, x_raw, grp[col].to_numpy(dtype=float))
                vals[grid < x_raw.min()] = np.nan
                vals[grid > x_raw.max()] = np.nan
                buffers[col][row] = vals

        result = pd.DataFrame({COL_CUSTOM_TIME: grid})
        for col in numeric_cols:
            result[col] = np.nanmean(buffers[col], axis=0)

        return result

    @staticmethod
    def compare_cohorts(
        df: pd.DataFrame,
        column: str,
        x_col: str = COL_CUSTOM_TIME,
        n_bins: int = 50,
    ) -> None:
        """
        Compare resigned vs retained cohorts for one column.
        Auto-dispatches: numeric → trajectory, categorical → grouped bar.

        Usage:
            Employee.compare_cohorts(df, COL_SALARY_EUROS)
            Employee.compare_cohorts(df, COL_ESTABLISHMENT)   # categorical
        """
        if not _is_numeric_col(column, df):
            Employee._cohorts_categorical(df, column)
            return

        resigned = Employee.cohort_average(df, has_resigned=True,  x_col=x_col, n_bins=n_bins)
        retained = Employee.cohort_average(df, has_resigned=False, x_col=x_col, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        _plot_series_on_ax(
            ax, [retained, resigned], x_col=COL_CUSTOM_TIME, y_col=column,
            labels=["Non-démissionnaires", "Démissionnaires"], show_markers=False,
        )
        ax.set_xlabel(x_col)
        fig.tight_layout()
        fig.show()

    @staticmethod
    def _cohorts_categorical(df: pd.DataFrame, column: str) -> None:
        per_emp = (
            df.groupby(COL_MATRICULE)[[COL_HAS_RESIGNED, column]]
            .agg({COL_HAS_RESIGNED: "max", column: lambda s: s.mode().iloc[0]})
            .reset_index()
        )

        categories = sorted(per_emp[column].dropna().unique())
        n_cats = len(categories)
        x = np.arange(n_cats)
        bar_width = 0.35
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(max(8, n_cats * 0.9), 4.5))

        for idx, (flag, cohort_name) in enumerate({0: "Non-démissionnaires", 1: "Démissionnaires"}.items()):
            cohort = per_emp[per_emp[COL_HAS_RESIGNED] == flag]
            counts = cohort[column].value_counts()
            total = counts.sum()
            props = [counts.get(cat, 0) / total * 100 for cat in categories]

            bars = ax.bar(
                x + (idx - 0.5) * bar_width, props, bar_width,
                label=cohort_name, color=cmap(idx), alpha=0.85,
            )
            for bar, pct in zip(bars, props):
                if pct > 1:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("% au sein de la cohorte")
        ax.set_title(f"{column} : En Non-démissionnaires vs Démissionnaires")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.show()

    @staticmethod
    def cohort_multiplot(
        df: pd.DataFrame,
        order_by: str = COL_CUSTOM_TIME,
        n_bins: int = 50,
        exclude_cols: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Grid of subplots comparing resigned vs retained cohorts for every numeric column.
        Uses true time so the x-axis reflects actual career duration.

        `df` must be the full raw employee DataFrame (with a 'label' column).
        `order_by` is the time column used to build the shared grid (default: COL_CUSTOM_TIME).

        Usage:
            Employee.cohort_multiplot(df)
            Employee.cohort_multiplot(df, order_by=COL_CUSTOM_TIME)
            Employee.cohort_multiplot(df, exclude_cols={COL_AGE_YEARS})
        """
        resigned = Employee.cohort_average(df, has_resigned=True,  x_col=order_by, n_bins=n_bins)
        retained = Employee.cohort_average(df, has_resigned=False, x_col=order_by, n_bins=n_bins)

        excluded = _MULTIPLOT_EXCLUDE | set(exclude_cols or [])
        y_cols = [c for c in resigned.columns if c != COL_CUSTOM_TIME and c not in excluded]

        ncols = 3
        nrows = ceil(len(y_cols) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

        for i, col in enumerate(y_cols):
            _plot_series_on_ax(axes.ravel()[i], [retained, resigned], x_col=COL_CUSTOM_TIME, y_col=col,
                               labels=["Non-démissionnaires", "Démissionnaires"], show_markers=False)

        for j in range(len(y_cols), len(axes.ravel())):
            axes.ravel()[j].set_visible(False)

        fig.suptitle("Toutes les colonnes : Non-démissionnaires vs Démissionnaires", y=1.02)
        fig.tight_layout()
        plt.show()
