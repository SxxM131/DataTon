"""Uni-D 배터리 열폭주 6단계 분류 — 17팀 최종 모델 공통 모듈.

구성
  1. 이미지 경로 연결 (베이스라인 resolve_image_paths 동작 유지)
  2. 이미지 색 통계 11개 (무지개 의사색상 → 의사 온도, 이미지마다 독립 계산)
  3. 테이블 피처 266개: 원시 센서·가스·환경·메타 + 이미지 색 통계 + 초 기준 과거 윈도우·변화량·임계 도달 후 경과초
  4. 규칙 피처 30개 (r_*): 라벨 규칙 역추적에서 얻은 상수 임계값 기반 인과 피처
  5. 압력 상승 피처 (q_*): 3단계(압력 연속 상승) 판정을 과거 관측만으로 근사 — 2·3단계 재판정 모델 전용
  6. LightGBM 다중 분류 2개 (결정적 설정): 기본 모델(296개 피처), 2·3단계 재판정 모델(296개 + 압력 상승 피처)
  7. pairdiff CNN: 현재 + 직전 프레임 + 차이 (9채널, 128×128, timm resnet18) → 6단계 확률
  8. 후기 융합: P = P_tab^(1-w) · P_cnn^w (w=0.2) → argmax + 1
  9. 2·3단계 재판정: 8의 결과가 2 또는 3인 행만, 재판정 모델(테이블 단독)의 p2·p3 중 큰 쪽으로 다시 정한다.
     나머지 단계(1, 4, 5, 6)의 예측은 건드리지 않는다.
     (2·3단계 경계는 열화상에 보이지 않아 이미지 확률을 섞지 않는다 — 검증에서도 섞지 않는 쪽이 같거나 나았다.)
 10. 5단계 근거 확인: 5단계 이상으로 나온 행은 그 구성의 5단계 이벤트가 한 번이라도 관측된 뒤에만 인정한다 (아니면 4단계).
     - 각형 28Ah 가열·각형 280Ah 과충전: 전압 급락 이후 온도 급상승
     - 원통형 4.8Ah·6Ah 가열: 첫 전압 급락 이후 두 번째 전압 강하
     모델이 '시간이 지났다'는 이유만으로 5단계로 넘기는 것을 막는다.
 11. 6단계 근거 확인: 6단계로 나온 행이라도 온도 합(t1+t2)의 현재 값이 지금까지의 최고점이면 5단계로 둔다
     (6단계는 온도 합의 최고점부터이므로, 아직 내려오기 시작하지 않았으면 최고점이 지났다고 볼 수 없다). 10번보다 먼저 적용한다.
 12. 원통형 4단계 최소 길이: 원통형에서 4단계에 들어선 뒤 30초가 지나기 전에는 5단계 이상으로 넘기지 않는다.
 13. 되돌아가지 않기: 같은 실험에서 앞선 시점에 4단계(또는 5단계) 이상으로 판정한 적이 있으면 그 아래로 내려가지 않는다
     (라벨은 97개 실험 모두 단계가 뒤로 가지 않는다). 10~12번 뒤에 적용한다. 3·6단계에는 적용하지 않는다(검증에서 손해).

모든 시계열 피처는 입력 frame에 있는 행만, 실험별로 relative_time_s 정렬 후 현재 이하 시각의 관측만 사용한다.
추론 대상 실험의 전체 통계·미래 관측·다른 실험 관측을 쓰지 않는다.
학습 데이터에서 산출해 저장하는 통계는 하나뿐이다: 배터리 구성(종류·용량·유발 방식)별 압력 변동 폭의 중앙값 (5번 피처의 단위 환산용).
→ 행 순서, 다른 실험 유무, 미래 행 유무와 무관하게 같은 행에는 같은 값이 나온다.

원본 실험 코드(이 파일은 import하지 않고 복사함):
  experiments/exp01_image_color.py (color_stats), experiments/tab/feats.py (build),
  experiments/rules/rules.py (rule_features), experiments/rules/run_rules.py (LightGBM 설정),
  experiments/stage3/presrise.py (pressure_features), experiments/stage3/run_presrise.py (2·3단계 재판정),
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
# 5. 압력 상승 피처 (experiments/stage3/presrise.py, 열 이름 q_*) — 2·3단계 재판정 모델 전용
#    3단계는 '표면 온도 50~90℃에서 압력이 연속으로 오르기 시작하는 지점'이다 (원본 데이터셋 구축 가이드, 공개 문서).
#    가이드의 정의는 '20초 뒤 압력'과 '실험 전체 압력 범위로 정규화한 값'을 쓰므로 그대로는 쓸 수 없다
#    (미래 관측·실험 전체 통계 금지). 여기서는 과거만으로 근사한다.
#      - 20초(5~30초) 전 대비 압력 상승량 p(t) − p(t−w). 온도 조건(50~90℃)은 w초 전 시점 기준과 현재 시점 기준 두 가지.
#      - 실험 전체 압력 범위 대신 '학습 실험에서 구한 같은 배터리 구성의 압력 범위 중앙값'으로 나눈다 (range_table).
#        학습 데이터에서만 산출한 전처리 통계이며, 추론 대상 실험의 값은 현재 이하 시각의 관측만 쓴다.
#      - 범위 추정이 틀릴 수 있어 임계값을 여러 개(0.01~0.08) 두고, 초기 압력 p0로 나눈 버전도 함께 둔다.
#      - 가이드에서 임계 규칙이 다른 구성(각형 280Ah 가열, 파우치 73Ah 가열 SOC 50%)은 피처를 비워 둔다 (NaN).
# ---------------------------------------------------------------------------
P_KEY = ['bat_class', 'bat_cc', 'tr_method']
P_THRS = (0.01, 0.02, 0.03, 0.04, 0.08)
P_LAGS = (5.0, 10.0, 15.0, 20.0, 30.0)


def _cfg_key(bat_class, bat_cc, tr_method) -> tuple:
    """배터리 구성 키 (종류, 용량, 유발 방식). 결측은 -1."""
    f = lambda x: -1.0 if pd.isna(x) else round(float(x), 3)  # noqa: E731
    return (f(bat_class), f(bat_cc), f(tr_method))


def pressure_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """학습 실험별 압력 범위(max − min)와 구성 키. 학습 데이터에만 호출한다 (index = experiment_id)."""
    g = frame.assign(experiment_id=frame['experiment_id'].astype(str)).groupby('experiment_id')
    out = g[P_KEY].first()
    out['rng'] = g['meas_p'].max() - g['meas_p'].min()
    return out


def range_table(exp_rng: pd.DataFrame) -> dict:
    """구성별 압력 범위 중앙값 (+ 배터리 종류별·전체 중앙값을 대체값으로). weights에 저장되는 유일한 학습 통계."""
    cfg = {_cfg_key(*k): float(v) for k, v in exp_rng.groupby(P_KEY)['rng'].median().items()}
    cls = {round(float(k), 3): float(v) for k, v in exp_rng.groupby('bat_class')['rng'].median().items()}
    return {'cfg': cfg, 'cls': cls, 'all': float(exp_rng['rng'].median())}


def lookup_range(table: dict, key: tuple) -> float:
    if key in table['cfg']:
        return table['cfg'][key]
    return table['cls'].get(key[0], table['all'])


def loo_ranges(exp_rng: pd.DataFrame) -> dict:
    """(학습용) 각 학습 실험에 대해 자기 자신을 뺀 같은 구성 실험들의 범위 중앙값 — 추론 때와 같은 조건을 만든다."""
    out = {}
    all_med = float(exp_rng['rng'].median())
    for e, r in exp_rng.iterrows():
        same = exp_rng[(exp_rng[P_KEY] == r[P_KEY]).all(axis=1)].drop(index=e)
        if len(same) == 0:
            same = exp_rng[exp_rng.bat_class == r.bat_class].drop(index=e)
        out[e] = float(same['rng'].median()) if len(same) else all_med
    return out


def doc_threshold(bat_class, bat_cc, tr_method, soc_status) -> float:
    """가이드의 구성별 기본 임계값: 각형 280Ah 가열, 파우치 73Ah 가열(SOC 50%)은 0.04, 나머지는 0.02."""
    special = (bat_class == 3 and bat_cc == 280 and tr_method == 1) or (bat_class == 2 and bat_cc == 73 and tr_method == 1 and soc_status == 1)
    return 0.04 if special else 0.02


def _pressure_one(t, p, t1, R, thr_doc=0.02) -> dict:
    """한 실험(시간순 정렬)의 압력 상승 피처. 모든 누적은 과거 방향(accumulate)."""
    n = len(t)
    out = {'q_R': np.full(n, R), 'q_doc_thr': np.full(n, thr_doc)}
    t1cm = np.fmax.accumulate(t1)
    gate50 = t1cm >= 50
    pmin = np.fmin.accumulate(p)
    out['q_level'] = (p - pmin) / R                                   # 지금까지의 최솟값 대비 압력 수준 (추정 범위 단위)
    run_rng = np.maximum(np.fmax.accumulate(p) - pmin, 1e-6)
    for w in P_LAGS:
        j = np.searchsorted(t, t - w, side='right') - 1               # t−w 이하의 마지막 관측
        ok = j >= 0
        d = np.zeros(n); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(n, np.nan); t1b[ok] = t1[j[ok]]
        valid = ok & (t1b >= 50) & (t1b <= 90)                        # w초 전 시점의 온도 조건 (w초 늦게 확인하는 셈)
        dr = np.where(valid, d / R, 0.0)
        k = int(w)
        out[f'q_d{k}'] = d / R
        out[f'q_d{k}_cm'] = np.fmax.accumulate(dr)                    # 조건을 만족한 상승량의 지금까지 최댓값
        valid_c = ok & (t1 >= 50) & (t1 <= 90)                        # 현재 온도 기준 (지연 없음)
        drc = np.where(valid_c, d / R, 0.0)
        out[f'q_d{k}_cmc'] = np.fmax.accumulate(drc)
        if w == 20.0:
            out['q_d20_runrng'] = np.where(valid, d / run_rng, 0.0)
            for th in P_THRS:
                first = np.minimum.accumulate(np.where(dr >= th, t, np.inf))
                out[f'q_since_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
                first = np.minimum.accumulate(np.where(drc >= th, t, np.inf))
                out[f'q_sincec_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
    # 초기 압력 p0 기준 버전 (실험 첫 관측 = 항상 과거)
    p0 = max(abs(p[0]), 1e-3) if n else 1.0
    out['q_p0'] = np.full(n, p[0] if n else np.nan)
    out['q_p_over_p0'] = p / p0
    for w in (10.0, 20.0, 30.0):
        j = np.searchsorted(t, t - w, side='right') - 1
        ok = j >= 0
        d = np.zeros(n); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(n, np.nan); t1b[ok] = t1[j[ok]]
        k = int(w)
        dl = np.where(ok & (t1b >= 50) & (t1b <= 90), d / p0, 0.0)
        dc = np.where(ok & (t1 >= 50) & (t1 <= 90), d / p0, 0.0)
        out[f'q_d{k}_p0cm'] = np.fmax.accumulate(dl)
        out[f'q_d{k}_p0cmc'] = np.fmax.accumulate(dc)
        if w == 20.0:
            for th in (0.01, 0.02, 0.03, 0.05):
                first = np.minimum.accumulate(np.where(dc >= th, t, np.inf))
                out[f'q_sincep0_{int(th*100):02d}'] = np.where(np.isfinite(first), t - first, -1.0)
    # 가이드 임계값(구성별 0.02 / 0.04) 기준: 누적 최대 상승량 ÷ 임계, 임계를 처음 넘은 뒤 경과 초
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


Q_COLS = list(_pressure_one(np.zeros(1), np.ones(1), np.full(1, 20.0), 1.0))


def pressure_features(frame: pd.DataFrame, table: dict, own_range: dict | None = None) -> pd.DataFrame:
    """압력 상승 피처 (index = frame.index). frame에 들어 있는 행만으로, 실험별·시간순으로 계산한다.

    table: range_table() 결과 (학습 데이터 통계). own_range: (학습 시에만) 실험별 '자기 자신을 뺀' 범위 추정값.
    """
    n = len(frame)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float) if c in frame.columns else np.full(n, np.nan)  # noqa: E731
    t_all, p_all, t1_all = num('relative_time_s'), num('meas_p'), num('meas_t1')
    meta = {c: num(c) for c in P_KEY + ['soc_status']}
    res = {k: np.full(n, np.nan) for k in Q_COLS}
    for e, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]                 # 시간순 (동률은 sample_id) → 입력 행 순서와 무관
        i0 = idx[0]
        bc, cc, tm, soc = (meta[c][i0] for c in P_KEY + ['soc_status'])
        thr = doc_threshold(bc, cc, tm, soc)
        if thr > 0.03:
            continue                                                  # 임계 규칙이 다른 구성은 비워 둔다 (NaN)
        R = own_range[e] if own_range is not None and e in own_range else lookup_range(table, _cfg_key(bc, cc, tm))
        with np.errstate(invalid='ignore', divide='ignore'):
            o = _pressure_one(t_all[idx], p_all[idx], t1_all[idx], max(float(R), 1e-6), thr)
        for k in Q_COLS:
            res[k][idx] = o[k]
    return pd.DataFrame(res, index=frame.index)


def redecide_23(pred: np.ndarray, p23: np.ndarray) -> np.ndarray:
    """pred가 2 또는 3인 행만 p23(재판정 모델 확률)의 p2·p3 중 큰 쪽으로 다시 정한다. 다른 단계는 그대로."""
    out = np.asarray(pred).copy()
    sel = (out == 2) | (out == 3)
    out[sel] = np.where(p23[sel, 2] > p23[sel, 1], 3, 2)
    return out


# ---------------------------------------------------------------------------
# 6. LightGBM (experiments/rules/run_rules.py: LGB_DEFAULT + fast 설정, random_state 42)
# ---------------------------------------------------------------------------
LGB_PARAMS = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1.0, n_jobs=4,
                  verbose=-1, deterministic=True, force_row_wise=True, random_state=SEED)


def build_model(**overrides):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **overrides})


# ---------------------------------------------------------------------------
# 7. pairdiff CNN (현재 프레임 + 직전 프레임 + 차이 = 9채널, 128×128)
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
# 8. 후기 융합 (experiments/img/latefuse.py 와 같은 식)
# ---------------------------------------------------------------------------
def fuse(p_tab: np.ndarray, p_cnn: np.ndarray, w: float = FUSE_W) -> np.ndarray:
    """P ∝ P_tab^(1-w) · P_cnn^w (로그 공간, 안정화 상수 포함) → 행마다 합 1로 정규화."""
    z = np.log(p_tab + 1e-9) * (1 - w) + np.log(p_cnn + 1e-6) * w
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# 9. 5단계 근거 확인 (experiments/robust/surge_gate.py)
#    가이드상 각형 28Ah 가열·각형 280Ah 과충전의 5단계는 '4단계(전압 급락) 이후 온도가 급상승하는 지점'이다.
#    학습 실험 23개 전부에서 아래 급상승 조건은 5단계 시작 +0~+2행에 처음 참이 되고, 4단계 동안에는 한 번도 참이 아니었다
#    (4단계 중 최대: 직전 관측 대비 0.80℃/s, 10초 상승 9.6℃).
#    행 모델은 4단계가 학습 실험보다 길어지면 급상승 없이도 5단계로 넘기는 경향이 있어 (4단계 구간을 3~5배로 늘린 합성 검증에서
#    4단계 행의 13~40%를 5단계로 예측), 급상승을 관측하기 전의 5단계 예측은 4단계로 되돌린다.
#    과거 관측과 고정 상수만 사용한다 (학습 통계 없음).
# ---------------------------------------------------------------------------
SURGE_CFGS = ((3, 28.0, 1), (3, 280.0, 2))     # (bat_class, bat_cc, tr_method)
SURGE_RISE = 12.0                              # 10초(이하 마지막 관측) 전 대비 온도 상승 ℃
SURGE_WINDOW_S = 10.0


def surge_seen(frame: pd.DataFrame) -> np.ndarray:
    """전압 급락 이벤트 이후 온도 급상승을 관측한 적이 있는가 (frame 행 위치 순서, bool).

    급상승 = 직전 관측 대비 초당 상승 ≥ T1_RATE_STAGE5(0.84℃/s) 또는 10초 이하 전 마지막 관측 대비 상승 ≥ SURGE_RISE.
    비교 시점도 전압 급락(직전 관측 대비 ≤ V_DROP 또는 V < V_LOW) 이후여야 한다. 실험별 시간순, 현재 이하 관측만 사용.
    """
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)  # noqa: E731
    t_all, t1_all, v_all = num('relative_time_s'), num('meas_t1'), num('meas_v')
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        t, t1, v = t_all[idx], t1_all[idx], v_all[idx]
        with np.errstate(invalid='ignore', divide='ignore'):
            dv = np.r_[0.0, np.diff(v)]
            ev = (np.nan_to_num(dv, nan=0.0) <= V_DROP) | (np.nan_to_num(v, nan=np.inf) < V_LOW)
            if not ev.any():
                continue
            ivd = int(np.argmax(ev))
            j = np.searchsorted(t, t - SURGE_WINDOW_S, side='right') - 1
            ok = j >= ivd
            rise = np.zeros(len(t)); rise[ok] = t1[ok] - t1[j[ok]]
            rate = np.r_[0.0, np.diff(t1) / np.diff(t)]
            rate[:ivd + 1] = 0.0
        hit = (np.nan_to_num(rise, nan=0.0) >= SURGE_RISE) | (np.nan_to_num(rate, nan=0.0) >= T1_RATE_STAGE5)
        out[idx] = np.maximum.accumulate(hit)
    return out


# 원통형(4.8Ah·6Ah 가열): 가이드상 5단계 = '4단계 이후 15초 안에 전압이 25% 이상 다시 떨어진 지점(하강 완료점)'.
#   학습 실험 31개 중 29개에서 아래 조건은 5단계 시작 행 또는 그 이전에 처음 참이 된다 (나머지 2개는 +1행, +2행 이상).
#   4단계 구간을 5배로 늘린 합성 검증: 조건 없이 4단계 행 1,077개를 5단계로 예측 → 조건을 넣으면 308개 (Macro F1 0.825 → 0.895).
#   공통 검증에서는 +0.0004 (바뀐 5행이 모두 정답).
VDD_CFGS = ((1, 4.8, 1), (1, 6.0, 1))          # (bat_class, bat_cc, tr_method)
VDD_THETA = 0.25                               # 직전 15초 최고 전압 대비 하강 비율
VDD_WINDOW_S = 15.0
VDD_LOW_V = 0.5                                # 첫 급락에서 이미 이 값 아래로 내려가면 '두 번째 강하'를 볼 수 없다 → 근거 있음으로 둔다


def second_drop_seen(frame: pd.DataFrame) -> np.ndarray:
    """첫 전압 급락 이후 두 번째 전압 강하를 관측한 적이 있는가 (frame 행 위치 순서, bool).

    두 번째 강하 = 현재 전압 ≤ (1 − VDD_THETA) × (직전 15초 동안, 첫 급락 이후 관측 중 최고 전압).
    첫 급락 이후 전압이 VDD_LOW_V 미만이면 그 시점부터 참으로 둔다. 실험별 시간순, 현재 이하 관측만 사용.
    """
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)  # noqa: E731
    t_all, v_all = num('relative_time_s'), num('meas_v')
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        t, v = t_all[idx], v_all[idx]
        with np.errstate(invalid='ignore'):
            dv = np.r_[0.0, np.diff(v)]
            ev = (np.nan_to_num(dv, nan=0.0) <= V_DROP) | (np.nan_to_num(v, nan=np.inf) < V_LOW)
        if not ev.any():
            continue
        ivd = int(np.argmax(ev))
        m = len(t)
        lo = np.maximum(np.searchsorted(t, t - VDD_WINDOW_S, side='left'), ivd)
        hit = np.zeros(m, dtype=bool)
        for k in range(ivd + 1, m):
            if lo[k] < k:
                ref = np.fmax.reduce(v[lo[k]:k])            # NaN은 건너뜀
                hit[k] = bool(v[k] <= (1.0 - VDD_THETA) * ref)
        hit |= (np.arange(m) >= ivd) & (np.nan_to_num(v, nan=np.inf) < VDD_LOW_V)
        out[idx] = np.maximum.accumulate(hit)
    return out


def _cfg_mask(frame: pd.DataFrame, cfgs) -> np.ndarray:
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)  # noqa: E731
    bc, cc, tm = num('bat_class'), num('bat_cc'), num('tr_method')
    mask = np.zeros(len(frame), dtype=bool)
    for a, b, c in cfgs:
        mask |= (bc == a) & (cc == b) & (tm == c)
    return mask


def gate_stage5(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """5단계 이상 예측인데 그 구성의 5단계 이벤트(온도 급상승 / 두 번째 전압 강하)가 아직 관측되지 않은 행을 4단계로 되돌린다.

    이 구성들의 학습 실험 54개는 모두 5단계를 거쳐 6단계로 간다 → 5단계 이벤트가 없으면 6단계일 수도 없다.
    (4단계 구간을 늘린 합성 검증에서 근거 없이 6단계로 건너뛰는 예측이 있었다.)
    """
    out = np.asarray(pred).copy()
    for cfgs, seen_fn in ((SURGE_CFGS, surge_seen), (VDD_CFGS, second_drop_seen)):
        target = _cfg_mask(frame, cfgs)
        if target.any():
            out[target & (out >= 5) & ~seen_fn(frame)] = 4
    return out


# 6단계: 가이드상 6단계 = 온도 합(t1+t2)이 최고점에 이른 시점부터.
#   현재 값이 지금까지의 최고점(같은 값 포함)이면, 최고점은 지금이거나 아직 오지 않았다 → 6단계로 두지 않는다.
#   공통 검증(3회)에서 이런 행을 6 → 5로 바꾸면 54행 중 37행이 정답이 되고 14행이 오답이 된다 (+0.0024, 짝지은 부트스트랩 개선 확률 1.00).
#   '초당 0.2℃ 넘게 오를 때만' 적용하는 변형도 검증 점수는 같았지만, 5단계가 2~3배 느리게 진행되는 합성 검증에서는
#   속도 조건이 없는 쪽이 조기 6단계 예측을 더 많이 막았다 (Macro F1 ×2: 0.9075 → 0.9124, ×3: 0.8883 → 0.8992).
#   더 강한 조건(최고점보다 2℃ 넘게 내려오거나 10초 넘게 정체해야 6단계)은 평상시 손해(−0.0008 이하)라 넣지 않는다.
#   반대 방향(최고점 갱신이 멈춘 지 오래면 6단계)은 중간 봉우리가 흔해서 크게 해롭다 (−0.01~−0.06) → 넣지 않는다.


def still_climbing(frame: pd.DataFrame) -> np.ndarray:
    """온도 합(t1+t2)의 현재 값이 지금까지의 최고점(같은 값 포함)인가 (frame 행 위치 순서, bool). 실험별 시간순, 현재 이하 관측만 사용."""
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)  # noqa: E731
    t_all, s_all = num('relative_time_s'), num('meas_t1') + num('meas_t2')
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        sm = s_all[idx]
        with np.errstate(invalid='ignore'):
            out[idx] = sm >= np.fmax.accumulate(sm) - 1e-9
    return out


def gate_stage6(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """6단계 예측인데 온도 합의 현재 값이 지금까지의 최고점인 행(아직 내려오기 시작하지 않음)을 5단계로 둔다. 다른 행은 그대로."""
    out = np.asarray(pred).copy()
    m = out == 6
    if m.any():
        out[m & still_climbing(frame)] = 5
    return out


# 원통형 4단계 최소 길이: 가이드의 전압 강하 판정은 '15초 창 + 복귀 여부 확인(10초 유지)'을 쓰므로, 첫 강하(4단계) 직후에는
#   다음 강하(5단계)가 곧바로 판정되지 않는다. 학습 원통형 31개 실험의 4단계는 모두 30초 이상이고(정확히 30초가 3개),
#   전압이 20초 만에 0V까지 내려간 실험도 5단계 라벨은 30초 시점에 시작한다.
#   행 모델은 첫 강하가 두 행에 걸쳐 이어질 때 이를 '두 번째 강하'로 보고 5단계를 일찍 내는 경우가 있다.
#   공통 검증(3회): 31행이 정답으로 바뀌고 오답으로 바뀐 행은 없다 (+0.0008, 개선 확률 0.95).
#   하한을 학습 fold에서만 구해도 +0.0006 (0.95). 다른 구성·다른 단계의 최소 길이 제약은 손해라 넣지 않는다.
STAGE4_MIN_S = 30.0


def floor_stage4(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """원통형(VDD_CFGS): 같은 실험에서 처음 4단계 이상으로 판정한 시각부터 STAGE4_MIN_S초가 지나기 전의 5단계 이상 판정을 4단계로 둔다."""
    out = np.asarray(pred).copy()
    target = _cfg_mask(frame, VDD_CFGS)
    if not target.any():
        return out
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    t_all = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    for _, idx in pd.Series(np.arange(len(frame))).groupby(eid, sort=False).indices.items():
        if not target[idx[0]]:
            continue
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        x = out[idx]
        if (x >= 4).any():
            t_in = t_all[idx][int(np.argmax(x >= 4))]
            out[idx] = np.where((x > 4) & (t_all[idx] - t_in < STAGE4_MIN_S - 1e-9), 4, x)
    return out


# 되돌아가지 않기: 라벨은 97개 실험 모두 단계가 뒤로 가지 않는다. 행 단위 모델은 4단계 이후에 가끔 낮은 단계로 되돌아간다.
#   공통 검증(3회): 4·5단계에 적용하면 +0.0057 (짝지은 부트스트랩 개선 확률 0.98, Test에 있는 구성만 +0.0020 / 0.96).
#   3단계(−0.0005)와 6단계(−0.0005)는 첫 판정이 이른 경우가 많아 굳히면 손해 → 적용하지 않는다.
#   근거 확인(위) 뒤에 적용하므로, 근거 없이 넘긴 5단계 판정이 굳어지지 않는다.
#   4단계를 굳히는 데는 전압 급락 이벤트 관측을 요구한다: 공통 검증에서는 결과가 완전히 같고(바뀌는 행 없음),
#   3단계가 2~3배 느리게 진행되는 합성 검증에서 근거 없이 한 번 나온 4단계가 굳어지는 것을 막는다.
HOLD_LEVELS = (4, 5)


def vdrop_seen(frame: pd.DataFrame) -> np.ndarray:
    """전압 급락 이벤트(직전 관측 대비 ≤ V_DROP 또는 V < V_LOW)를 관측한 적이 있는가 (frame 행 위치 순서, bool). 과거 관측만 사용."""
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)  # noqa: E731
    t_all, v_all = num('relative_time_s'), num('meas_v')
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        v = v_all[idx]
        with np.errstate(invalid='ignore'):
            dv = np.r_[0.0, np.diff(v)]
            ev = (np.nan_to_num(dv, nan=0.0) <= V_DROP) | (np.nan_to_num(v, nan=np.inf) < V_LOW)
        out[idx] = np.maximum.accumulate(ev)
    return out


def hold_stage(pred: np.ndarray, frame: pd.DataFrame, levels=HOLD_LEVELS) -> np.ndarray:
    """같은 실험의 앞선 시점(현재 포함 이전 행)에 s단계 이상으로 판정한 적이 있으면 s 미만으로 내려가지 않게 한다 (s ∈ levels).

    s = 4는 전압 급락 이벤트를 관측한 뒤부터만 적용한다 (근거 없는 4단계 판정은 굳히지 않는다).
    """
    out = np.asarray(pred).copy()
    n = len(frame)
    eid = frame['experiment_id'].astype(str).to_numpy()
    sid = frame['sample_id'].astype(str).to_numpy()
    t_all = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    vd = vdrop_seen(frame)
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        x = out[idx]
        for lv in levels:
            seen = np.maximum.accumulate(x >= lv)
            if lv == 4:
                seen = seen & vd[idx]
            x = np.where(seen & (x < lv), lv, x)
        out[idx] = x
    return out


def apply_rules(pred: np.ndarray, p_tab23: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """최종 라벨 보정 순서: 2·3단계 재판정 → 6단계 근거 확인 → 5단계 근거 확인 → 원통형 4단계 최소 길이 → 4·5단계 되돌아가지 않기."""
    x = gate_stage5(gate_stage6(redecide_23(pred, p_tab23), frame), frame)
    return hold_stage(floor_stage4(x, frame), frame)


def to_label(p: np.ndarray) -> np.ndarray:
    return (np.argmax(p, 1) + 1).astype(int)
