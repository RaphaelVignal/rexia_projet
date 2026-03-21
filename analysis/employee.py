from typing import Optional, Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analysis.utils import (
    plot_two_columns_by_x,
    plot_n_series_by_x,
    multiplot_all_columns_by_anciennete,
)

COL_MATRICULE = 'matricule'
COL_SALARY_EUROS = 'Salaire (Euros)'
COL_VEHICLE = 'Véhicule'
COL_HIERARCHY_LEVEL = 'Niveau hiérarchique'
COL_AGE_YEARS = 'Âge (années)'
COL_ESTABLISHMENT = 'Etablissement'
COL_SENIORITY_GROUP_YEARS = 'Ancienneté groupe (années)'
COL_CONTRACT_START_YEARS = 'Début de contrat (années)'
COL_LAST_RAISE_MONTHS = 'Dernière augmentation (mois)'
COL_LAST_PROMOTION_MONTHS = 'Dernière promotion (mois)'
COL_JOB_FAMILY = "Famille d'emploi"
COL_HAS_RESIGNED = "label"
COL_CUSTOM_TIME = "Temps (années)"

COLUMN_DTYPES = {
    COL_MATRICULE: int,
    COL_HIERARCHY_LEVEL: int,
    COL_AGE_YEARS: int,
    COL_VEHICLE: int,
    COL_ESTABLISHMENT: str,
    COL_JOB_FAMILY: str,
    COL_SALARY_EUROS: float,
    COL_SENIORITY_GROUP_YEARS: float,
    COL_CONTRACT_START_YEARS: float,
    COL_LAST_RAISE_MONTHS: float,
    COL_LAST_PROMOTION_MONTHS: float,
}


def fix_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    for column, dtype in COLUMN_DTYPES.items():
        df[column] = df[column].astype(dtype, errors='raise')
    return df


class Employee:
    """Load by matricule, perform deeper analysis and comparison."""

    def __init__(self, df: pd.DataFrame, matricule: int, name: str | None = None):
        self.df = df[df[COL_MATRICULE] == matricule].copy()
        self.df = fix_dataframe_types(self.df)
        self.matricule: int = matricule
        self.name: str | None = name
        self.has_resigned = self.df[COL_HAS_RESIGNED].sum() > 0

        self.df.sort_values(by=[COL_MATRICULE], inplace=True, ascending=False)
        self.time_points = len(self.df)

        start_at: float = self.df[COL_LAST_RAISE_MONTHS].min()
        end_at: float = self.df[COL_LAST_RAISE_MONTHS].max()

        # Add a normalised time column: years since first observation (not seniority).
        if self.time_points <= 1:
            self.df[COL_CUSTOM_TIME] = 0.0
        else:
            span_months = end_at - start_at
            self.df[COL_CUSTOM_TIME] = [
                i * span_months / ((self.time_points - 1) * 12.0)
                for i in range(self.time_points)
            ]

    @staticmethod
    def get_random(df: pd.DataFrame, has_resigned: bool | None = None, name: str | None = None) -> 'Employee':
        candidates: pd.DataFrame
        if has_resigned is None:
            candidates = df[COL_MATRICULE]
        else:
            candidates = df.loc[df[COL_HAS_RESIGNED] == int(has_resigned), COL_MATRICULE]

        if candidates.empty:
            raise ValueError("No employee matches the requested resignation filter.")

        random_matricule = candidates.sample(n=1).iloc[0]
        return Employee(df, random_matricule, name=name)

    def _label(self) -> str:
        return self.name if self.name is not None else f"Matricule {self.matricule}"

    def determine_type(self, col: str) -> str:
        return "line" if is_numeric_dtype(self.df[col].dropna()) else "scatter"

    def compare_columns_over_time(self, y1: str, y2: str, x: str = COL_SENIORITY_GROUP_YEARS) -> None:
        if y1 not in self.df.columns or y2 not in self.df.columns:
            raise ValueError(f"Columns '{y1}' and/or '{y2}' not found in dataframe.")

        fig, ax1, ax2 = plot_two_columns_by_x(
            df=self.df,
            x_col=x,
            y1_col=y1,
            y2_col=y2,
            y1_kind=self.determine_type(y1),
            y2_kind=self.determine_type(y2),
        )
        fig.show()

    # ------------------------------------------------------------------
    # Multi-employee comparative plots
    # ------------------------------------------------------------------

    def compare(
        self,
        others: 'Employee | list[Employee]',
        column: str,
        x: str = COL_SENIORITY_GROUP_YEARS,
        show_resignations: bool = False,
    ) -> None:
        """
        Superpose one column from self and one or more other employees on a single axis.
        """
        if isinstance(others, Employee):
            others = [others]

        all_employees = [self, *others]
        fig, ax = plot_n_series_by_x(
            dataframes=[e.df for e in all_employees],
            x_col=x,
            y_col=column,
            labels=[e._label() for e in all_employees],
            kinds=[e.determine_type(column) for e in all_employees],
            show_resignations=show_resignations,
        )
        fig.show()

    def multiplot(
        self,
        x: str = COL_SENIORITY_GROUP_YEARS,
        exclude_cols: Optional[Iterable[str]] = None,
        show_resignations: bool = False,
        others: 'Employee | list[Employee] | None' = None,
    ) -> None:
        """
        Grid of every column vs x, with self and any others superposed per subplot.
        """
        if others is None:
            others = []
        elif isinstance(others, Employee):
            others = [others]

        excluded = {COL_MATRICULE} | set(exclude_cols or [])

        all_employees = [self, *others]
        multiplot_all_columns_by_anciennete(
            dataframes=[e.df for e in all_employees],
            x_col=x,
            exclude_cols=excluded,
            labels=[e._label() for e in all_employees] if others else None,
            kinds=None,
            show_resignations=show_resignations,
        )

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def show_timeline(self, centered: bool = True) -> None:
        # If centered, start at 0 instead of the employee seniority.
        # Should show a hline when employee resigns, until we find another contract start.
        pass