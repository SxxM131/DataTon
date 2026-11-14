"""실험 02: 로지스틱 회귀 → GBDT(HistGradientBoosting) + 피처 그룹 추가 효과.

피처 그룹 (모두 과거 방향 / 행 단위):
  base     : relative_time_s + 센서 5개 + 30초 윈도우(mean/max/delta)
  color    : 의사 온도 이미지 통계 11개 (exp01)
  imgtime  : 의사 온도 30초 윈도우 + 실험 첫 프레임 대비 변화량
  extra    : 가스 10종 + atmp_t, thermal_t, indu_t(+결측 플래그) + 배터리 메타 4개 + thermal_t/CO/CO2 윈도우
  cum      : 현재까지 누적 최댓값/최솟값 (라벨 단조 증가 성질 활용)

예) python experiments/exp02_gbdt.py --data_dir ~/data/battery_data_light --cache_dir <캐시 폴더>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as M  # noqa: E402
from exp01_image_color import load, first_frame_delta  # noqa: E402

GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
ENV = ['atmp_t', 'thermal_t', 'indu_t']
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
IMG_KEY = ['pt_mean', 'pt_p90', 'pt_p99', 'pt_frac50', 'pt_frac70', 'white_frac']
CUM_MAX = ['meas_t1', 'meas_t2', 'meas_p', 'thermal_t', 'pt_p90', 'pt_p99', 'pt_frac50', 'pt_frac70', 'white_frac', 'meas_co']
CUM_MIN = ['meas_v']


def cumulative(frame: pd.DataFrame, base: pd.DataFrame, max_cols, min_cols) -> pd.DataFrame:
    """실험별 시간순 누적 최댓값/최솟값 (현재 시점 포함, 과거만). (실험, 시각) 중복이 없음을 확인함."""
    t = base['relative_time_s'].to_numpy(dtype=float)
    out = pd.DataFrame(index=base.index)
    cols = {f'{c}_cummax': (c, np.fmax) for c in max_cols if c in base} | {f'{c}_cummin': (c, np.fmin) for c in min_cols}
    arrs = {k: np.full(len(base), np.nan) for k in cols}
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        for k, (c, fn) in cols.items():
            arrs[k][idx] = fn.accumulate(base[c].to_numpy(dtype=float)[idx])
    for k in cols:
        out[k] = arrs[k]
    return out


def make_X(frame, color, groups):
    base = pd.DataFrame({'experiment_id': frame.experiment_id, 'relative_time_s': pd.to_numeric(frame.relative_time_s)})
    base[M.SENSORS] = M.numeric(frame, M.SENSORS)
    base[GAS + ENV + META] = M.numeric(frame, GAS + ENV + META)
    c = color.reindex(frame.sample_id.values).set_index(frame.index)
    base[list(c.columns)] = c
    parts = [base[['relative_time_s'] + M.SENSORS].drop(columns='meas_a'),
             M.add_causal_features(base, [s for s in M.SENSORS if s != 'meas_a'])]
    if 'color' in groups:
        parts.append(c)
    if 'imgtime' in groups:
        parts += [M.add_causal_features(base, IMG_KEY), first_frame_delta(frame, c, IMG_KEY)]
    if 'extra' in groups:
        parts += [base[GAS + ENV + META], base['indu_t'].isna().astype(float).rename('indu_t_missing'),
                  M.add_causal_features(base, ['thermal_t', 'meas_co', 'meas_co2'])]
    if 'cum' in groups:
        mx = [k for k in CUM_MAX if ('extra' in groups or k not in ('thermal_t', 'meas_co'))
              and ('color' in groups or not k.startswith(('pt_', 'white')))]
        parts.append(cumulative(frame, base, mx, CUM_MIN))
    return pd.concat(parts, axis=1)


def gbdt():
    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, max_features=0.8, early_stopping=False,
                                          random_state=M.SEED)


def mf1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--cache_dir', type=Path, required=True)
    args = ap.parse_args()
    tr, _, ctr = load(args.data_dir, 'train', args.cache_dir)
    va, _, cva = load(args.data_dir, 'validation', args.cache_dir)
    ytr, yva = tr.label.astype(int), va.label.astype(int)

    configs = [
        ('base', []),
        ('base+color', ['color']),
        ('base+color+imgtime', ['color', 'imgtime']),
        ('base+color+cum', ['color', 'cum']),
        ('base+color+imgtime+cum', ['color', 'imgtime', 'cum']),
        ('base+extra', ['extra']),
        ('base+color+extra', ['color', 'extra']),
        ('base+color+imgtime+extra+cum', ['color', 'imgtime', 'extra', 'cum']),
    ]
    for name, groups in configs:
        Xtr, Xva = make_X(tr, ctr, groups), make_X(va, cva, groups)
        cv = [mf1(ytr.iloc[b], gbdt().fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b]))
              for a, b in GroupKFold(5).split(Xtr, ytr, tr.experiment_id)]
        p = gbdt().fit(Xtr, ytr).predict(Xva)
        f1s = f1_score(yva, p, labels=M.LABELS, average=None)
        print(f'{name:30s} n_feat={Xtr.shape[1]:3d} | CV {np.mean(cv):.4f}±{np.std(cv):.4f} | Val {mf1(yva, p):.4f} | '
              f'per-class {np.round(f1s, 3).tolist()}', flush=True)


if __name__ == '__main__':
    main()
