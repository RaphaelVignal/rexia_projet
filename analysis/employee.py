import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

from analysis.utils import plot_column_vs_anciennete, plot_two_columns_by_x, plot_same_column_two_dataframes_by_x

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
    """ load by matricule, perform deeper analysis and comparison """
    def __init__(self, df: pd.DataFrame, matricule: int):
        self.df = df[df[COL_MATRICULE] == matricule].copy()
        self.df = fix_dataframe_types(self.df)
        self.matricule: int = matricule

        self.df.sort_values(by=[COL_MATRICULE], inplace=True, ascending=False)
        self.time_points = len(self.df)

    def determine_type(self, col: str) -> str:
        return "line" if is_numeric_dtype(self.df[col].dropna()) else "scatter"

    def compare_columns_over_time(self, y1: str, y2: str) -> None:
        if y1 not in self.df.columns or y2 not in self.df.columns:
            raise ValueError(f"Columns '{y1}' and/or '{y2}' not found in dataframe.")


        fig, ax1, ax2 = plot_two_columns_by_x(
            df=self.df,
            x_col=COL_SENIORITY_GROUP_YEARS,
            y1_col=y1,
            y2_col=y2,
            y1_kind=self.determine_type(y1),
            y2_kind=self.determine_type(y2),
        )
        fig.show()

    def compare_with_other(self, other: 'Employee', column: str) -> None:
        fig, ax = plot_same_column_two_dataframes_by_x(
            df1=self.df,
            df2=other.df,
            x_col=COL_SENIORITY_GROUP_YEARS,
            y_col=column,
            df1_label=f"Matricule {self.matricule}",
            df2_label=f"Matricule {other.matricule}",
            df1_kind=self.determine_type(column),
            df2_kind=other.determine_type(column),
        )
        fig.show()







