"""세션 A (센서/테이블 트랙) — 벡터화된 인과(causal) 피처 생성기.

원칙 (CLAUDE.md test.py 재현성 규칙):
- 입력 frame에 있는 행만 사용. 실험별로 relative_time_s 정렬 후 현재 시점 이하(<= t)의 관측만 사용.
- 윈도우·변화율은 전부 '초' 기준 (행 수 기준 아님) → 5초 / 10·15초 간격이 섞여도 의미가 같도록.
- 실험 첫 관측(시간상 항상 과거)과의 차이는 허용. 실험 전체 통계(평균·최댓값·길이·순위)는 사용하지 않음.
- 학습 통계(스케일러 등) 없음 → 같은 입력이면 항상 같은 결과, 행 순서·다른 실험 유무와 무관.

사용:
    X = build(frame, color, cfg)   # frame: 원본 CSV 행 (sample_id, experiment_id, relative_time_s, 센서...), color: sample_id 인덱스 이미지 통계
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p']          # meas_a는 항상 0 → 제외
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
ENV = ['atmp_t', 'thermal_t', 'indu_t']                      # indu_t.1은 Test에서 100% 결측 → 사용 금지
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
COLOR = ['pt_mean', 'pt_std', 'pt_p50', 'pt_p90', 'pt_p99', 'pt_max',
         'pt_frac20', 'pt_frac50', 'pt_frac70', 'pt_frac90', 'white_frac']
IMG_SIG = ['pt_mean', 'pt_p90', 'pt_frac50']                 # 시계열 피처를 만들 이미지 신호

# 기본 설정 = 'sensor5' (센서 5개 + 이미지만). extra=True면 가스·환경·메타 추가.
DEFAULT = dict(
    windows=(30, 120, 300),        # 초
    win_stats=('mean', 'max', 'min'),
    std_windows=(120,),
    delta_windows=(15, 30, 60, 120, 300),
    cum=True,                       # 누적 max/min, 첫 관측 대비, 최고점 이후 경과 시간
    thresh=True,                    # 임계값 최초 도달 이후 경과 시간 (t1>=45 등)
    img=True,                       # 이미지 색 통계 + 이미지 신호 시계열
    extra=False,                    # 가스 10종 + atmp_t/thermal_t/indu_t + 메타 4개
)


def _prep(frame: pd.DataFrame, color: pd.DataFrame | None, cols: list[str]):
    """실험·시간 기준 정렬된 numpy 배열들을 준비. order: 정렬 순서(원래 위치), 결과는 원래 순서로 되돌린다."""
    eid = frame['experiment_id'].astype(str).to_numpy()
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    codes = pd.factorize(eid, sort=True)[0]
    order = np.lexsort((t, codes))                         # 실험 → 시간 순 (안정 정렬)
    data = frame[[c for c in cols if c in frame.columns]].apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    if color is not None:
        c = color.reindex(frame['sample_id'].astype(str).values)
        for k in c.columns:
            data[k] = c[k].to_numpy()
    df = data.iloc[order].reset_index(drop=True)
    df.insert(0, '_t', t[order])
    df.insert(0, '_g', codes[order])
    return df, order


def _lag_index(g: np.ndarray, t: np.ndarray, w: float) -> np.ndarray:
    """각 행에 대해 같은 실험에서 시각 <= t-w 인 마지막 관측 위치 (없으면 -1)."""
    key = g * 1e7 + t
    j = np.searchsorted(key, key - w, side='right') - 1
    ok = (j >= 0) & (g[np.clip(j, 0, None)] == g)
    return np.where(ok, j, -1)


def _since(g, t, cond: np.ndarray) -> np.ndarray:
    """조건이 처음 참이 된 시각 이후 경과 초 (아직이면 -1). 과거 방향만."""
    first_t = pd.Series(np.where(cond, t, np.nan)).groupby(g).cummin().to_numpy()   # 지금까지 처음 참이 된 시각
    return np.where(np.isnan(first_t), -1.0, t - first_t)


def build(frame: pd.DataFrame, color: pd.DataFrame | None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT, **(cfg or {})}
    use_img = cfg['img'] and color is not None
    raw = SENS + (GAS + ENV + META if cfg['extra'] else [])
    df, order = _prep(frame, color if use_img else None, raw)
    g, t = df['_g'].to_numpy(), df['_t'].to_numpy()
    sig = SENS + (IMG_SIG if use_img else []) + (['thermal_t', 'meas_co', 'meas_co2', 'indu_t'] if cfg['extra'] else [])
    out = {'relative_time_s': t}
    for c in raw:
        out[c] = df[c].to_numpy()
    if use_img:
        for c in COLOR:
            out[c] = df[c].to_numpy()
    if cfg['extra']:
        out['indu_t_missing'] = df['indu_t'].isna().to_numpy(dtype=float)

    # --- 초 기준 롤링 윈도우: (t-W, t] (pandas time-offset rolling, 실험별) ---
    td = pd.to_timedelta(t, unit='s')
    sdf = df[sig].copy()
    sdf.index = td
    grp = sdf.groupby(g, sort=False)
    for w in cfg['windows']:
        r = grp.rolling(f'{w}s', min_periods=1)
        for st in cfg['win_stats']:
            v = getattr(r, st)().to_numpy()
            for k, s in enumerate(sig):
                out[f'{s}_{st}{w}s'] = v[:, k]
    for w in cfg['std_windows']:
        v = grp.rolling(f'{w}s', min_periods=2).std().to_numpy()
        for k, s in enumerate(sig):
            out[f'{s}_std{w}s'] = v[:, k]

    # --- 변화량·변화율 (초당): t-W 이하 마지막 관측과 비교 ---
    X = df[sig].to_numpy()
    for w in cfg['delta_windows']:
        j = _lag_index(g, t, w)
        ok = j >= 0
        jj = np.clip(j, 0, None)
        dt = np.where(ok, t - t[jj], np.nan)
        d = np.where(ok[:, None], X - X[jj], np.nan)
        for k, s in enumerate(sig):
            out[f'{s}_d{w}s'] = d[:, k]
            out[f'{s}_r{w}s'] = d[:, k] / dt            # 초당 변화율 (간격이 달라도 같은 단위)

    # --- 누적 피처 (라벨 단조 증가 → 지금까지의 극값이 단계 정보를 담음) ---
    if cfg['cum']:
        cs = pd.DataFrame(X, columns=sig)
        cg = cs.groupby(g, sort=False)
        cmax, cmin = cg.cummax().to_numpy(), cg.cummin().to_numpy()
        first = cg.transform('first').to_numpy()        # 실험 첫 관측 (항상 과거)
        for k, s in enumerate(sig):
            out[f'{s}_cummax'] = cmax[:, k]
            out[f'{s}_cummin'] = cmin[:, k]
            out[f'{s}_d0'] = X[:, k] - first[:, k]
            out[f'{s}_fromcmax'] = X[:, k] - cmax[:, k]
            if s != 'meas_v':
                # 누적 최댓값을 기록한 시각 이후 경과 초 (정점 이후 하강 단계 식별)
                at = pd.Series(np.where(X[:, k] >= cmax[:, k], t, np.nan)).groupby(g).ffill().to_numpy()
                out[f'{s}_since_cmax'] = t - at
        v, v0 = X[:, 0], first[:, 0]
        vmax = cmax[:, 0]
        out['v_ratio0'] = v / np.where(np.abs(v0) > 1e-6, v0, np.nan)
        out['v_ratio_max'] = v / np.where(vmax > 1e-6, vmax, np.nan)
        out['v_min_ratio_max'] = cmin[:, 0] / np.where(vmax > 1e-6, vmax, np.nan)

    # --- 임계값 최초 도달 이후 경과 시간 (과거 방향 이벤트 시계) ---
    if cfg['thresh']:
        t1, t2, v = df['meas_t1'].to_numpy(), df['meas_t2'].to_numpy(), df['meas_v'].to_numpy()
        vmax = pd.Series(v).groupby(g).cummax().to_numpy()
        for th in (45, 60, 80, 100, 150, 300):
            out[f'since_t1_{th}'] = _since(g, t, t1 >= th)
        for th in (100, 200):
            out[f'since_t2_{th}'] = _since(g, t, t2 >= th)
        for fr in (0.9, 0.5, 0.1):
            out[f'since_v_{fr}'] = _since(g, t, v < fr * vmax)
        out['since_v_zero'] = _since(g, t, v < 0.05)

    res = pd.DataFrame(out)
    res.index = frame.index[order]
    return res.loc[frame.index]
