"""세션 D — 라벨 규칙 역추적 결과를 인과적(causal) 피처로 만든 모듈.

역추적 결과 요약 (자세한 근거는 NOTES.md)
  2단계: meas_t1 >= 45 첫 행 (97/97 정확).
  3단계: meas_t1 >= 50 은 필요조건. 두 번째 조건은 센서 임계로 재현 불가 (압력 상승 시작 0~3행 '전'에 라벨이 바뀜).
  4단계: 전압 급락 이벤트 (직전 관측 대비 -0.56V 이하, 또는 v < 2.5). 정확 66/94, 라벨이 1행 빠른 경우 19/94.
  5단계: 배터리 종류별. bat1: v <= 1.355 / bat2: v == 0 / bat3·cc28: t1 상승률 >= 0.84℃/s / bat3·cc280: 불명확(thermal_t >= 170).
  6단계: (t1+t2)의 실험 전체 최댓값 시점 (비인과). 인과적으로는 '지금까지 최댓값에서 내려옴'으로 근사.

원칙 (CLAUDE.md test.py 재현성 규칙)
  - 입력 frame의 행만 사용. 실험별 relative_time_s 정렬 후 현재와 과거 행만 사용.
  - 차분은 '직전 관측 행' 기준, 변화율은 초당, 경과 시간은 초 → 5초 / 10·15초 간격이 섞여도 의미가 같다.
  - 학습된 통계 없음 (상수 임계값만). 행 순서, 다른 실험 유무, 미래 행 유무와 무관하게 같은 값.
  - 라벨, indu_t.1 은 쓰지 않는다. bat_class / bat_cc 는 행마다 주어지는 실험 상수(메타 컬럼)로만 사용.

사용
    R = rule_features(frame)            # index = frame.index, 열 이름은 'r_' 로 시작
    pred = rule_predict(frame)          # 순수 규칙 기반 단계(1~6) 예측, index = frame.index
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 역추적으로 얻은 상수 (라벨 있는 97개 실험에서만 도출)
T1_STAGE2 = 45.0          # 2단계: t1 >= 45
T1_STAGE3_MIN = 50.0      # 3단계 필요조건: t1 >= 50
T1_STAGE3_RULE = 55.0     # 순수 규칙 예측용 3단계 근사 임계 (rule_predict의 t3 인자로 바꿀 수 있음)
V_DROP = -0.56            # 4단계: 직전 관측 대비 전압 변화 <= -0.56 V
V_LOW = 2.5               # 4단계: 또는 v < 2.5 V
V_DROP2 = -0.4            # 2차 전압 강하 (bat1의 5단계 후보)
V_STAGE5_BAT1 = 1.355     # 5단계 bat1: v <= 1.355
T1_RATE_STAGE5 = 0.84     # 5단계 bat3/cc28: t1 상승률 >= 0.84 ℃/s (5초에 4.2℃)
TH_STAGE5_BAT3L = 170.0   # 5단계 bat3/cc280: thermal_t >= 170 (약한 규칙)
S_PEAK_MIN = 300.0        # 6단계: 지금까지 (t1+t2) 최댓값이 300 이상이고
S_PEAK_DROP = 5.0         #        현재 값이 그 최댓값보다 5 넘게 내려옴


def _sorted(frame: pd.DataFrame):
    eid = frame['experiment_id'].astype(str).to_numpy()
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    codes = pd.factorize(eid, sort=True)[0]
    order = np.lexsort((t, codes))                      # 실험 → 시간 (안정 정렬)
    return codes[order], t[order], order


def _num(frame, col, order):
    if col not in frame.columns:
        return np.full(len(order), np.nan)
    return pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)[order]


def _since(g, t, cond):
    """조건이 (같은 실험에서) 처음 참이 된 시각 이후 경과 초. 아직이면 -1."""
    first_t = pd.Series(np.where(cond, t, np.nan)).groupby(g).cummin().to_numpy()
    return np.where(np.isnan(first_t), -1.0, t - first_t)


def _cum(x, g, how):
    s = pd.Series(x).groupby(g)
    return (s.cummax() if how == 'max' else s.cummin()).to_numpy()


def _prev(x, g):
    return pd.Series(x).groupby(g).shift(1).to_numpy()


def _lag(g, t, x, w):
    """시각 <= t-w 인 마지막 관측의 값 (없으면 NaN). 초 기준."""
    key = g * 1e7 + t
    j = np.searchsorted(key, key - w, side='right') - 1
    ok = (j >= 0) & (g[np.clip(j, 0, None)] == g)
    return np.where(ok, x[np.clip(j, 0, None)], np.nan)


def rule_features(frame: pd.DataFrame, t3: float = T1_STAGE3_RULE) -> pd.DataFrame:
    """인과적 규칙 피처. 반환: index = frame.index 인 DataFrame (열 이름 'r_*')."""
    g, t, order = _sorted(frame)
    v, t1, t2 = _num(frame, 'meas_v', order), _num(frame, 'meas_t1', order), _num(frame, 'meas_t2', order)
    p, th = _num(frame, 'meas_p', order), _num(frame, 'thermal_t', order)
    bat, cc = _num(frame, 'bat_class', order), _num(frame, 'bat_cc', order)
    with np.errstate(invalid='ignore', divide='ignore'):
        dt = t - _prev(t, g)                             # 직전 관측과의 간격(초). 첫 행은 NaN
        dv = v - _prev(v, g)
        r_t1 = (t1 - _prev(t1, g)) / dt
        S = t1 + t2
        r_S = (S - _prev(S, g)) / dt
        vmax = _cum(v, g, 'max')
        Sc = _cum(S, g, 'max')
        out = {}

        # --- 2단계 / 3단계 필요조건 ---
        c2 = np.nan_to_num(t1, nan=-np.inf) >= T1_STAGE2
        c50 = np.nan_to_num(t1, nan=-np.inf) >= T1_STAGE3_MIN
        out['r_since_t1_45'] = _since(g, t, c2)
        out['r_since_t1_50'] = _since(g, t, c50)
        out['r_t1_over50'] = _cum(t1, g, 'max') - T1_STAGE3_MIN
        # 3단계 단서: 압력이 오르기 시작 (라벨은 그 0~3행 전에 바뀜)
        pmin = _cum(p, g, 'min')
        e2 = out['r_since_t1_45'] >= 0
        first2 = pd.Series(np.where(c2, p, np.nan))
        first2 = first2.where(pd.Series(c2).groupby(g).cumsum().to_numpy() == 1)   # 2단계 조건이 처음 참인 행의 압력만 남김
        p_at2 = first2.groupby(g).ffill().to_numpy()
        out['r_p_from_stage2'] = p - p_at2
        out['r_p_from_cummin'] = p - pmin
        for w in (30, 60):
            out[f'r_p_d{w}s'] = p - _lag(g, t, p, w)
        out['r_p_rate'] = (p - _prev(p, g)) / dt
        c_prise = e2 & (np.nan_to_num(p - _lag(g, t, p, 30), nan=0.0) >= 1.0)
        out['r_since_p_rise'] = _since(g, t, c_prise)

        # --- 4단계: 전압 급락 이벤트 ---
        c_drop = np.nan_to_num(dv, nan=0.0) <= V_DROP
        c4 = c_drop | (np.nan_to_num(v, nan=np.inf) < V_LOW)
        s4 = _since(g, t, c4)
        e4 = s4 >= 0
        out['r_since_vdrop'] = s4
        out['r_n_vdrop'] = pd.Series((np.nan_to_num(dv, nan=0.0) <= V_DROP2).astype(float)).groupby(g).cumsum().to_numpy()
        out['r_v_from_max'] = v - vmax
        out['r_v_dprev'] = dv

        # --- 5단계: 배터리 종류별 ---
        c_v1 = e4 & (np.nan_to_num(v, nan=np.inf) <= V_STAGE5_BAT1)
        c_v0 = np.nan_to_num(v, nan=np.inf) <= 0.0
        c_rate = e4 & (np.nan_to_num(r_t1, nan=0.0) >= T1_RATE_STAGE5)
        c_th = e4 & (np.nan_to_num(th, nan=-np.inf) >= TH_STAGE5_BAT3L)
        out['r_since_v_1p355'] = _since(g, t, c_v1)
        out['r_since_v_zero'] = _since(g, t, c_v0)
        out['r_since_t1_fast'] = _since(g, t, c_rate)
        out['r_since_th170'] = _since(g, t, c_th)
        out['r_t1_rate'] = r_t1
        out['r_t1_rate_cmax4'] = _cum(np.where(e4, np.nan_to_num(r_t1, nan=0.0), 0.0), g, 'max')   # 4단계 이벤트 이후 최대 상승률
        c5 = np.where(bat == 1, c_v1, np.where(bat == 2, c_v0, np.where(cc == 28, c_rate, c_th | c_rate)))
        c5 = c5.astype(bool)
        s5 = _since(g, t, c5)
        out['r_since_stage5_rule'] = s5

        # --- 6단계: (t1+t2) 피크 통과 ---
        at_max = np.nan_to_num(S, nan=-np.inf) >= np.nan_to_num(Sc, nan=np.inf)
        last_max_t = pd.Series(np.where(at_max, t, np.nan)).groupby(g).ffill().to_numpy()
        out['r_S'] = S
        out['r_S_cummax'] = Sc
        out['r_S_from_cummax'] = S - Sc
        out['r_S_since_cummax'] = t - last_max_t         # 최댓값을 마지막으로 갱신한 뒤 경과 초
        out['r_S_rate'] = r_S
        for w in (30, 60):
            out[f'r_S_d{w}s'] = S - _lag(g, t, S, w)
        c6 = e4 & (np.nan_to_num(Sc, nan=0.0) >= S_PEAK_MIN) & (np.nan_to_num(S - Sc, nan=0.0) < -S_PEAK_DROP)
        s6 = _since(g, t, c6)
        out['r_since_peak_passed'] = s6

        # --- 순수 규칙 기반 단계 (조건이 발화한 가장 높은 단계) ---
        c3 = np.nan_to_num(t1, nan=-np.inf) >= t3
        s3 = _since(g, t, c3)
        stage = np.ones(len(t))
        stage = np.where(e2, 2, stage)
        stage = np.where(s3 >= 0, 3, stage)
        stage = np.where(e4, 4, stage)
        stage = np.where(s5 >= 0, 5, stage)
        stage = np.where(s6 >= 0, 6, stage)
        out['r_stage'] = stage.astype(float)
        out['r_n_fired'] = (e2.astype(float) + (out['r_since_t1_50'] >= 0) + e4 + (s5 >= 0) + (s6 >= 0))

    res = pd.DataFrame(out)
    res.index = frame.index[order]
    return res.loc[frame.index]


def rule_predict(frame: pd.DataFrame, t3: float = T1_STAGE3_RULE) -> pd.Series:
    """순수 규칙 기반 단계 예측 (1~6 정수)."""
    return rule_features(frame, t3=t3)['r_stage'].astype(int)


FEATURE_NAMES = None  # rule_features(...).columns 로 확인


# ---------------------------------------------------------------------------
# 인과성 자체 점검: 행 섞기 / 실험 나눠 넣기 / 미래 행 빼기(앞 60%)에서 값이 완전히 같은지
# ---------------------------------------------------------------------------
def self_test(frame: pd.DataFrame, seed: int = 0) -> dict:
    base = rule_features(frame)
    rng = np.random.default_rng(seed)
    res = {}
    sh = frame.iloc[rng.permutation(len(frame))]
    res['shuffle'] = bool(rule_features(sh).loc[frame.index].equals(base))
    ok = True
    for e, sub in frame.groupby('experiment_id', sort=False):
        ok &= rule_features(sub).equals(base.loc[sub.index])
    res['split_by_experiment'] = bool(ok)
    t = frame['relative_time_s']
    rk = t.groupby(frame['experiment_id']).rank(method='first')
    n = frame.groupby('experiment_id')['relative_time_s'].transform('size')
    head = frame[rk <= np.ceil(0.6 * n)]
    res['drop_future_60pct'] = bool(rule_features(head).equals(base.loc[head.index]))
    return res


if __name__ == '__main__':
    import sys
    import time
    from pathlib import Path
    data = Path.home() / 'data/battery_data_light'
    te = pd.read_csv(data / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    t0 = time.time()
    R = rule_features(te)
    print(f'test {len(te)}행 / {te.experiment_id.nunique()}개 실험: 피처 {R.shape[1]}개, {1000 * (time.time() - t0):.0f} ms')
    print('자체 점검(test):', self_test(te))
    fired = pd.DataFrame({'t1>=45': R.r_since_t1_45 >= 0, 't1>=50': R.r_since_t1_50 >= 0, 'vdrop(4)': R.r_since_vdrop >= 0,
                          'stage5_rule': R.r_since_stage5_rule >= 0, 'peak_passed(6)': R.r_since_peak_passed >= 0})
    print('조건이 발화한 test 실험 수 (총 %d):' % te.experiment_id.nunique())
    print(fired.groupby(te.experiment_id.to_numpy()).any().sum().to_string())
    print('규칙 예측 단계 분포(test):', R.r_stage.astype(int).value_counts().sort_index().to_dict())
