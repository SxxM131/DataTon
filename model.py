"""Uni-D 배터리 열폭주 6단계 분류 — 17팀 최종 모델 공통 모듈.

구성
  1. 이미지 경로 연결 (베이스라인 resolve_image_paths 동작 유지)
  2. 이미지 색 통계 11개 (무지개 의사색상 → 의사 온도, 이미지마다 독립 계산)
  3. 테이블 피처 266개: 원시 센서·가스·환경·메타 + 이미지 색 통계 + 초 기준 과거 윈도우·변화량·임계 도달 후 경과초
  4. 규칙 피처 30개 (r_*): 라벨 규칙 역추적에서 얻은 상수 임계값 기반 인과 피처
  5. LightGBM 다중 분류 (결정적 설정)
  6. pairdiff CNN: 현재 + 직전 프레임 + 차이 (9채널, 128×128, timm resnet18) → 6단계 확률
  7. 후기 융합: P = P_tab^(1-w) · P_cnn^w (w=0.2) → argmax + 1

모든 시계열 피처는 입력 frame에 있는 행만, 실험별로 relative_time_s 정렬 후 현재 이하 시각의 관측만 사용한다.
학습된 통계(스케일러 등)는 없고, 실험 전체 통계·미래 관측·다른 실험 관측을 쓰지 않는다.
→ 행 순서, 다른 실험 유무, 미래 행 유무와 무관하게 같은 행에는 같은 값이 나온다.

원본 실험 코드(이 파일은 import하지 않고 복사함):
  experiments/exp01_image_color.py (color_stats), experiments/tab/feats.py (build),
  experiments/rules/rules.py (rule_features), experiments/rules/run_rules.py (LightGBM 설정),
  experiments/img/finetune2.py, experiments/img/infer_pair.py, experiments/img/infer_bench.py (CNN)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

SEED = 42
LABELS = list(range(1, 7))
N_JOBS = 6                                     # 공유 서버 CPU 경합 → 최대 6

# ---------------------------------------------------------------------------
# 1. 이미지 경로 연결 (베이스라인 v7과 동일)
# ---------------------------------------------------------------------------
def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    """CSV의 image_path를 실제 파일 경로로 바꿉니다.

    roots를 순서대로 기준 폴더로 시도하고, 모두 실패하면 파일명으로 찾습니다.
    """
    rel = frame['image_path'].astype(str)
    for root in roots:
        candidate = rel.map(lambda p: Path(root) / p)
        if candidate.map(Path.is_file).all():
            return candidate
    # 경로 규칙이 맞지 않으면 파일명 기준으로 검색합니다.
    index = {}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file():
                index.setdefault(path.name, path)
    candidate = rel.map(lambda p: index.get(Path(p).name))
    missing = candidate.isna().sum()
    assert missing == 0, f'이미지 {missing}개를 찾지 못했습니다. image_root를 확인하십시오. 예: {rel[candidate.isna()].iloc[0]}'
    return candidate


# ---------------------------------------------------------------------------
# 2. 이미지 색 통계 (이미지마다 독립 → 다른 행 정보 사용 없음)
#    열화상은 무지개 의사색상(마젠타=저온 → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색=고온).
#    색조(hue)를 의사 온도 pt = clip((300 - hue°)/300, 0, 1) 로 되돌린다 (흰색 = 1). 고정 상수만 사용.
# ---------------------------------------------------------------------------
COLOR_STATS = ['pt_mean', 'pt_std', 'pt_p50', 'pt_p90', 'pt_p99', 'pt_max',
               'pt_frac20', 'pt_frac50', 'pt_frac70', 'pt_frac90', 'white_frac']


def color_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        hsv = np.asarray(rgb.convert('HSV'), dtype=np.float32)
    h = hsv[..., 0] * (360.0 / 256.0)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    white = (s < 0.16) & (v > 0.78)
    valid = (v > 0.15) | white                      # 너무 어두운 픽셀(테두리 등)은 제외
    pt = np.clip((300.0 - h) / 300.0, 0.0, 1.0)     # 마젠타(300°)=0 ... 빨강(0°)=1
    pt = np.where(h > 330.0, 1.0, pt)               # 330~360°는 빨강 쪽으로 간주
    pt = np.where(white, 1.0, pt)
    x = pt[valid] if valid.any() else pt.ravel()
    return [float(x.mean()), float(x.std()), *map(float, np.percentile(x, [50, 90, 99])), float(x.max()),
            float((x >= .2).mean()), float((x >= .5).mean()), float((x >= .7).mean()), float((x >= .9).mean()),
            float(white.mean())]


def extract_image_features(frame: pd.DataFrame, paths, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """sample_id를 인덱스로 하는 이미지 색 통계 표 (행마다 독립 계산)."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(color_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=COLOR_STATS, index=pd.Index(frame['sample_id'].astype(str).values, name='sample_id'))


# ---------------------------------------------------------------------------
# 3. 테이블 피처 (experiments/tab/feats.py 의 build 그대로)
#    - 윈도우·변화율은 전부 '초' 기준 (Test의 10·15초 간격과 학습의 5초 간격이 섞여도 의미가 같도록)
#    - 최종 설정: extra=True(가스·환경·메타 포함), cum=False(누적 극값 피처 제외) → 266개
# ---------------------------------------------------------------------------
SENS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p']          # meas_a는 항상 0 → 제외
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
ENV = ['atmp_t', 'thermal_t', 'indu_t']                      # indu_t.1은 Test에서 100% 결측 → 사용 금지
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
IMG_SIG = ['pt_mean', 'pt_p90', 'pt_frac50']                 # 시계열 피처를 만들 이미지 신호

TAB_DEFAULT = dict(
    windows=(30, 120, 300),        # 초
    win_stats=('mean', 'max', 'min'),
    std_windows=(120,),
    delta_windows=(15, 30, 60, 120, 300),
    cum=True,                       # 누적 max/min, 첫 관측 대비, 최고점 이후 경과 시간
    thresh=True,                    # 임계값 최초 도달 이후 경과 시간 (t1>=45 등)
    img=True,                       # 이미지 색 통계 + 이미지 신호 시계열
    rule=False,                     # (사용 안 함) 단순 규칙형 피처
    extra=False,                    # 가스 10종 + atmp_t/thermal_t/indu_t + 메타 4개
)
TAB_CFG = {'extra': True, 'cum': False}                      # 최종 설정


def _sort_order(frame: pd.DataFrame):
    """실험 → 시간 순 정렬 순서. 동률(같은 실험·같은 시각)은 sample_id로 고정해 입력 순서와 무관하게 만든다."""
    eid = frame['experiment_id'].astype(str).to_numpy()
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    sid = frame['sample_id'].astype(str).to_numpy()
    codes = pd.factorize(eid, sort=True)[0]
    order = np.lexsort((sid, t, codes))
    return codes, t, order


def _prep(frame: pd.DataFrame, color: pd.DataFrame | None, cols: list[str]):
    """실험·시간 기준 정렬된 표. order: 정렬 순서(원래 위치), 결과는 원래 순서로 되돌린다."""
    codes, t, order = _sort_order(frame)
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
    """조건이 (같은 실험에서) 처음 참이 된 시각 이후 경과 초 (아직이면 -1). 과거 방향만."""
    first_t = pd.Series(np.where(cond, t, np.nan)).groupby(g).cummin().to_numpy()
    return np.where(np.isnan(first_t), -1.0, t - first_t)


def tab_features(frame: pd.DataFrame, color: pd.DataFrame | None, cfg: dict | None = None) -> pd.DataFrame:
    """frame: 원본 CSV 행, color: sample_id 인덱스 이미지 색 통계. 반환 index = frame.index."""
    cfg = {**TAB_DEFAULT, **(cfg or {})}
    use_img = cfg['img'] and color is not None
    raw = SENS + (GAS + ENV + META if cfg['extra'] else [])
    df, order = _prep(frame, color if use_img else None, raw)
    g, t = df['_g'].to_numpy(), df['_t'].to_numpy()
    sig = SENS + (IMG_SIG if use_img else []) + (['thermal_t', 'meas_co', 'meas_co2', 'indu_t'] if cfg['extra'] else [])
    out = {'relative_time_s': t}
    for c in raw:
        out[c] = df[c].to_numpy()
    if use_img:
        for c in COLOR_STATS:
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

    # --- 누적 피처 (최종 모델에서는 cum=False로 사용하지 않음; 실험 코드와의 호환을 위해 유지) ---
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

    if cfg['rule']:
        t1 = df['meas_t1'].to_numpy()
        t1max = pd.Series(t1).groupby(g).cummax().to_numpy()
        out['t1_cummax'] = t1max
        for th in (45, 50):
            out[f'rule_t1max_ge{th}'] = (t1max >= th).astype(float)
            out[f'rule_t1_ge{th}'] = (t1 >= th).astype(float)
            out[f'rule_since_t1_{th}'] = _since(g, t, t1 >= th)
        out['rule_t1_m50'] = t1 - 50.0
        out['rule_t1max_m50'] = t1max - 50.0

    res = pd.DataFrame(out)
    res.index = frame.index[order]
    return res.loc[frame.index]


# ---------------------------------------------------------------------------
# 4. 규칙 피처 (experiments/rules/rules.py 의 rule_features 그대로, 열 이름 r_*)
#    라벨 있는 97개 실험의 역추적에서 얻은 상수 임계값만 사용 (학습 통계 없음).
#    차분은 직전 관측 행 기준, 변화율은 초당, 경과 시간은 초.
# ---------------------------------------------------------------------------
T1_STAGE2 = 45.0          # 2단계: t1 >= 45
T1_STAGE3_MIN = 50.0      # 3단계 필요조건: t1 >= 50
T1_STAGE3_RULE = 55.0     # 규칙 단계 r_stage 용 3단계 근사 임계
V_DROP = -0.56            # 4단계: 직전 관측 대비 전압 변화 <= -0.56 V
V_LOW = 2.5               # 4단계: 또는 v < 2.5 V
V_DROP2 = -0.4            # 2차 전압 강하 횟수
V_STAGE5_BAT1 = 1.355     # 5단계 bat1: v <= 1.355
T1_RATE_STAGE5 = 0.84     # 5단계 bat3/cc28: t1 상승률 >= 0.84 ℃/s
TH_STAGE5_BAT3L = 170.0   # 5단계 bat3/cc280: thermal_t >= 170 (약한 규칙)
S_PEAK_MIN = 300.0        # 6단계: 지금까지 (t1+t2) 최댓값이 300 이상이고
S_PEAK_DROP = 5.0         #        현재 값이 그 최댓값보다 5 넘게 내려옴


def _num(frame, col, order):
    if col not in frame.columns:
        return np.full(len(order), np.nan)
    return pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)[order]


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
    codes, t_all, order = _sort_order(frame)
    g, t = codes[order], t_all[order]
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
        # 3단계 단서: 압력이 오르기 시작
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

        # --- 6단계: (t1+t2) 지금까지의 최고점 통과 (인과 근사) ---
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

        # --- 규칙 기반 단계 (조건이 발화한 가장 높은 단계) ---
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


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """최종 테이블 입력 = 테이블 피처 266개 + 규칙 피처 30개 (index = frame.index).

    frame에 들어 있는 행만으로 만든다. image_features: sample_id 인덱스 색 통계 (extract_image_features).
    """
    return pd.concat([tab_features(frame, image_features, TAB_CFG), rule_features(frame)], axis=1)


# ---------------------------------------------------------------------------
# 5. LightGBM (experiments/rules/run_rules.py: LGB_DEFAULT + fast 설정, random_state 42)
# ---------------------------------------------------------------------------
LGB_PARAMS = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1.0, n_jobs=4,
                  verbose=-1, deterministic=True, force_row_wise=True, random_state=SEED)


def build_model(**overrides):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **overrides})


# ---------------------------------------------------------------------------
# 6. pairdiff CNN (현재 프레임 + 직전 프레임 + 차이 = 9채널, 128×128)
#    직전 프레임 = 입력에 있는 같은 실험의 행 중 시간상 바로 앞 관측 (실험 첫 행은 자기 자신 → 차이 0).
#    추론은 1장씩 + TF32 off + cudnn 결정적 → 배치 구성·행 순서와 무관하게 비트 단위 동일.
# ---------------------------------------------------------------------------
IMG_SIZE = 128
CNN_ARCH = 'resnet18'                 # 학습 시 사전학습 가중치: timm 'resnet18.tv_in1k'
CNN_PRETRAINED = 'resnet18.tv_in1k'
FUSE_W = 0.2                          # 후기 융합 이미지 가중치 (고정)
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def decode_image(path) -> np.ndarray:
    """RGB로 읽어 128×128 bilinear 리사이즈 (종횡비 무시), uint8 (H, W, 3)."""
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)


def decode_images(paths, n_jobs: int = N_JOBS) -> np.ndarray:
    return np.stack(Parallel(n_jobs=n_jobs, batch_size=64)(delayed(decode_image)(p) for p in paths))


def set_torch_deterministic():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_cnn(pretrained: bool = False):
    """CNN 구조. pretrained=False면 네트워크 접근 없이 구조만 만든다 (추론용)."""
    import torch.nn as nn
    import timm

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(CNN_PRETRAINED if pretrained else CNN_ARCH, pretrained=pretrained,
                                              num_classes=0, in_chans=9)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(512, 6)

        def forward(self, x):
            return self.head(self.drop(self.backbone(x)))

    return Net()


def load_cnn(path, device):
    """fp16으로 저장된 state_dict를 float32 모델에 올린다."""
    import torch
    m = make_cnn(pretrained=False)
    sd = torch.load(path, map_location='cpu')
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return m.to(device).eval()


def prev_index(frame: pd.DataFrame) -> np.ndarray:
    """각 행(위치 기준)의 직전 관측 위치. 같은 실험 안에서 relative_time_s 순 (동률은 sample_id), 첫 행은 자기 자신."""
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    sid = frame['sample_id'].to_numpy().astype(str)
    prev = np.arange(len(frame))
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t[idx]))]
        prev[idx[1:]] = idx[:-1]
    return prev


def cnn_probs(frame: pd.DataFrame, images: np.ndarray, model, device) -> np.ndarray:
    """frame 행 순서(위치)대로 6단계 확률. images: frame과 같은 순서의 (N, 128, 128, 3) uint8. 1장씩 추론."""
    import torch
    set_torch_deterministic()
    frame = frame.reset_index(drop=True)
    prev = prev_index(frame)
    imgs = torch.from_numpy(images)
    mean = torch.tensor(IMNET_MEAN * 2, device=device).view(1, 6, 1, 1)
    std = torch.tensor(IMNET_STD * 2, device=device).view(1, 6, 1, 1)
    out = np.zeros((len(frame), 6), np.float32)
    with torch.no_grad():
        for i in range(len(frame)):              # 배치 1: 배치 구성에 따른 수치 차이 제거
            x = torch.cat([imgs[[i]], imgs[[prev[i]]]], -1).to(device).permute(0, 3, 1, 2).float().div(255)
            x = torch.cat([(x - mean) / std, (x[:, :3] - x[:, 3:]) * 2], 1)
            out[i] = torch.softmax(model(x).float(), 1).cpu().numpy()[0]
    return out


# ---------------------------------------------------------------------------
# 7. 후기 융합 (experiments/img/latefuse.py 와 같은 식)
# ---------------------------------------------------------------------------
def fuse(p_tab: np.ndarray, p_cnn: np.ndarray, w: float = FUSE_W) -> np.ndarray:
    """P ∝ P_tab^(1-w) · P_cnn^w (로그 공간, 안정화 상수 포함) → 행마다 합 1로 정규화."""
    z = np.log(p_tab + 1e-9) * (1 - w) + np.log(p_cnn + 1e-6) * w
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def to_label(p: np.ndarray) -> np.ndarray:
    return (np.argmax(p, 1) + 1).astype(int)
