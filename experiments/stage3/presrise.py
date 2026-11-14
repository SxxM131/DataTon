"""3단계 '압력 연속 상승' 이벤트의 인과적 근사 피처.

원본 데이터셋 구축 가이드의 정의 (공개 문서):
  3단계 = 표면 온도 50~90℃ 이면서, 정규화(실험 전체 min-max)한 압력이 20초 뒤에 0.02 이상 오르는 첫 지점
  (각형 280Ah 가열·파우치 73Ah 가열 SOC 50%는 0.04 → 없으면 0.03 → 없으면 0.02)
원래 정의는 '20초 뒤 값'과 '실험 전체 압력 범위'를 쓰므로 그대로는 쓸 수 없다 (미래 관측·실험 전체 통계 금지).
여기서는 과거만 사용한다:
  - 20초 전 대비 압력 상승량 p(t) − p(t−20s) 을, 20초 전 시점의 온도가 50~90℃일 때만 인정 (20초 늦게 확인하는 셈)
  - 실험 전체 압력 범위 대신 '학습 실험에서 구한 같은 배터리 구성의 압력 범위 중앙값'(range_table)을 쓴다
    → 학습 데이터에서만 산출한 전처리 통계이며, 추론 대상 실험의 미래·전체 통계는 쓰지 않는다
  - 범위 추정이 틀릴 수 있으므로 임계값을 여러 배수(0.01~0.08)로 둔다
"""
import numpy as np
import pandas as pd

KEY = ['bat_class', 'bat_cc', 'tr_method']
THRS = (0.01, 0.02, 0.03, 0.04, 0.08)
LAGS = (5.0, 10.0, 15.0, 20.0, 30.0)


def experiment_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """학습 실험별 압력 범위(max−min)와 구성 키. 학습 데이터에만 호출한다."""
    g = frame.groupby('experiment_id')
    out = g[KEY].first()
    out['rng'] = g['meas_p'].max() - g['meas_p'].min()
    return out


def range_table(exp_rng: pd.DataFrame) -> dict:
    """구성별 압력 범위 중앙값 (+ 배터리 종류별, 전체 중앙값을 대체값으로)."""
    return {'cfg': exp_rng.groupby(KEY)['rng'].median().to_dict(),
            'cls': exp_rng.groupby('bat_class')['rng'].median().to_dict(),
            'all': float(exp_rng['rng'].median())}


def lookup(table: dict, key) -> float:
    k = tuple(key)
    if k in table['cfg']:
        return float(table['cfg'][k])
    return float(table['cls'].get(k[0], table['all']))


def doc_threshold(bat_class, bat_cc, tr_method, soc_status) -> float:
    """가이드의 구성별 기본 임계값: 각형 280Ah 가열, 파우치 73Ah 가열(SOC 50%)은 0.04, 나머지는 0.02."""
    special = (bat_class == 3 and bat_cc == 280 and tr_method == 1) or (bat_class == 2 and bat_cc == 73 and tr_method == 1 and soc_status == 1)
    return 0.04 if special else 0.02


def _one(t, p, t1, R, thr_doc=0.02):
    n = len(t)
    out = {'q_R': np.full(n, R), 'q_doc_thr': np.full(n, thr_doc)}
    t1cm = np.maximum.accumulate(t1)
    gate50 = t1cm >= 50
    pmin = np.minimum.accumulate(p)
    out['q_level'] = (p - pmin) / R                                   # 지금까지의 최솟값 대비 압력 수준 (추정 범위 단위)
    run_rng = np.maximum(np.maximum.accumulate(p) - pmin, 1e-6)
    for w in LAGS:
        j = np.searchsorted(t, t - w, side='right') - 1               # t−w 이하의 마지막 관측
        ok = j >= 0
        d = np.zeros(n); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(n, np.nan); t1b[ok] = t1[j[ok]]
        valid = ok & (t1b >= 50) & (t1b <= 90)                        # 이벤트 지점(t−w)의 온도 조건
        dr = np.where(valid, d / R, 0.0)
        k = int(w)
        out[f'q_d{k}'] = d / R
        out[f'q_d{k}_cm'] = np.maximum.accumulate(dr)                 # 조건을 만족한 상승량의 지금까지 최댓값
        # 현재 온도 기준 버전: '지난 w초 동안 이미 올랐다면 앞으로도 오른다'고 보고 지금 시점을 이벤트로 인정 (지연 없음)
        valid_c = ok & (t1 >= 50) & (t1 <= 90)
        drc = np.where(valid_c, d / R, 0.0)
        out[f'q_d{k}_cmc'] = np.maximum.accumulate(drc)
        if w == 20.0:
            out['q_d20_runrng'] = np.where(valid, d / run_rng, 0.0)
            for th in THRS:
                first = np.minimum.accumulate(np.where(dr >= th, t, np.inf))
                out[f'q_since_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
                first = np.minimum.accumulate(np.where(drc >= th, t, np.inf))
                out[f'q_sincec_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
    # 초기 압력 p0 기준 버전: 원형 4.8Ah처럼 실험 끝에 압력이 0 근처로 떨어지는 경우 전체 범위 ≈ 최대 압력 ≈ p0의 1.1~2.4배
    p0 = max(abs(p[0]), 1e-3)
    out['q_p0'] = np.full(n, p[0])
    out['q_p_over_p0'] = p / p0
    for w in (10.0, 20.0, 30.0):
        j = np.searchsorted(t, t - w, side='right') - 1
        ok = j >= 0
        d = np.zeros(n); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(n, np.nan); t1b[ok] = t1[j[ok]]
        k = int(w)
        dl = np.where(ok & (t1b >= 50) & (t1b <= 90), d / p0, 0.0)          # 20초 늦게 확인
        dc = np.where(ok & (t1 >= 50) & (t1 <= 90), d / p0, 0.0)            # 현재 온도 기준
        out[f'q_d{k}_p0cm'] = np.maximum.accumulate(dl)
        out[f'q_d{k}_p0cmc'] = np.maximum.accumulate(dc)
        if w == 20.0:
            for th in (0.01, 0.02, 0.03, 0.05):
                first = np.minimum.accumulate(np.where(dc >= th, t, np.inf))
                out[f'q_sincep0_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
    # 문서 임계값(구성별 0.02 / 0.04) 기준: 누적 최대 상승량 ÷ 임계, 임계를 처음 넘은 뒤 경과 시간
    for nm in ('q_d20_cm', 'q_d20_cmc'):
        out[nm + '_doc'] = out[nm] / thr_doc
    for nm, src in (('q_since_doc', 'q_d20_cm'), ('q_sincec_doc', 'q_d20_cmc')):
        first = np.minimum.accumulate(np.where(out[src] >= thr_doc, t, np.inf))
        out[nm] = np.where(np.isfinite(first), t - first, -1.0)
    out['q_d20_p0cmc_doc'] = out['q_d20_p0cmc'] / (thr_doc * 1.5)
    out['q_t1_in'] = ((t1 >= 50) & (t1 <= 90)).astype(float)
    t50 = t[gate50][0] if gate50.any() else np.inf
    out['q_since_t50'] = np.where(gate50, t - t50, -1.0)
    return out


def features(frame: pd.DataFrame, table: dict, own_range: dict | None = None, mask_special: bool = False) -> pd.DataFrame:
    """frame의 행만으로 계산. own_range: (학습 시) 실험별로 '자기 자신을 뺀' 범위 추정값을 줄 때 사용."""
    t_all = frame['relative_time_s'].to_numpy(float)
    res = None
    for e, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        f = frame.iloc[idx]
        R = own_range[e] if own_range is not None and e in own_range else lookup(table, f[KEY].iloc[0])
        m = f[KEY + ['soc_status']].iloc[0]
        o = _one(t_all[idx], f['meas_p'].to_numpy(float), f['meas_t1'].to_numpy(float), max(float(R), 1e-6),
                 doc_threshold(m['bat_class'], m['bat_cc'], m['tr_method'], m['soc_status']))
        if res is None:
            res = {k: np.full(len(frame), np.nan) for k in o}
        if mask_special and o['q_doc_thr'][0] > 0.03:
            continue                                   # 임계 규칙이 다른 구성(0.04→0.03→0.02)은 피처를 비워 둔다 (NaN)
        for k, v in o.items():
            res[k][idx] = v
    return pd.DataFrame(res, index=frame.index)


def loo_ranges(exp_rng: pd.DataFrame) -> dict:
    """학습 실험 각각에 대해, 자기 자신을 뺀 같은 구성 실험들의 범위 중앙값 (학습 시 과신 방지)."""
    out = {}
    all_med = float(exp_rng['rng'].median())
    for e, r in exp_rng.iterrows():
        same = exp_rng[(exp_rng[KEY] == r[KEY]).all(axis=1)].drop(index=e)
        if len(same) == 0:
            same = exp_rng[exp_rng.bat_class == r.bat_class].drop(index=e)
        out[e] = float(same['rng'].median()) if len(same) else all_med
    return out
