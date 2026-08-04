"""
데이터 로드 및 기본 확인 (노트북 1절).
"""
import pandas as pd

from .config import DATA_PATH


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(df.shape)
    return df


def basic_info(df):
    df.info()
    print("\n결측치 개수:\n", df.isna().sum().sum())
    print("\n중복 행 개수:", df.duplicated().sum())
    return df.describe().T
