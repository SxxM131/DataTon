"""세션 D — 분석용 공통 로더 (라벨 규칙 역추적)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common'), str(HERE.parent / 'tab')]
import cv  # noqa
DATA = Path.home() / 'data/battery_data_light'
CACHE = Path.home() / 'jaehyeok/cache/rules'
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']


def load():
    p = CACHE / 'pool.pkl'
    if p.exists():
        return pd.read_pickle(p)
    pool = cv.load_pool(DATA)
    pool = pool.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').reset_index()
    pool = pool.rename(columns={'index': 'orig_idx'})
    pool['row'] = pool.groupby('experiment_id').cumcount()
    pool.to_pickle(p)
    return pool


def onsets(pool, k):
    """단계 k의 첫 행 (실험별). 반환: experiment_id -> row 번호"""
    d = pool[pool.label == k].groupby('experiment_id')['row'].min()
    return d
