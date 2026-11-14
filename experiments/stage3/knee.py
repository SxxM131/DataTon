"""무릎점(knee) 검출 피처: 신호가 '자기 자신의 최근 직선 추세'에서 벗어나는 정도를 실험별 잡음 대비 z-점수로 표현.

3단계 시작 전후로 압력이 계단처럼 뛰거나(280Ah), 평평하다가 오르기 시작하거나(소형 셀), 온도 상승이 가팔라지는(73Ah 과충전) 등
실험마다 다른 형태의 '꺾임'이 보인다. 느린 직선 상승(드리프트)에는 반응하지 않고 꺾임에만 반응하도록,
과거 구간(t-gap-ref, t-gap]에 직선을 맞춰 현재로 외삽한 값과 최근 평균의 차이를 본다.
모든 계산은 같은 실험의 현재·과거 행만 사용하고, 윈도우는 초 단위다.
"""
import numpy as np
import pandas as pd


def _csum(x):
    return np.concatenate([[0.0], np.cumsum(x)])


def knee_z(t, x, recent_s=15.0, gap_s=15.0, ref_s=180.0, min_ref=6):
    """한 실험(시간순 t, 신호 x). 반환: z, resid(원 단위), sigma. 결측은 0."""
    n = len(t)
    xv = np.where(np.isnan(x), 0.0, x)
    ok = (~np.isnan(x)).astype(float)
    c1, cx, ct, ctt, ctx = _csum(ok), _csum(xv), _csum(t * ok), _csum(t * t * ok), _csum(t * xv)
    i1 = np.arange(1, n + 1)
    a0 = np.searchsorted(t, t - recent_s, side='right')            # 최근 구간 (t-recent, t]
    r1 = np.searchsorted(t, t - gap_s, side='right')               # 기준 구간 (t-gap-ref, t-gap]
    r0 = np.searchsorted(t, t - gap_s - ref_s, side='right')
    nr = c1[r1] - c1[r0]
    na = c1[i1] - c1[a0]
    with np.errstate(invalid='ignore', divide='ignore'):
        sx, st, stt, stx = cx[r1] - cx[r0], ct[r1] - ct[r0], ctt[r1] - ctt[r0], ctx[r1] - ctx[r0]
        den = nr * stt - st * st
        slope = np.where(den > 1e-9, (nr * stx - st * sx) / den, 0.0)
        icpt = (sx - slope * st) / nr
        ta = (ct[i1] - ct[a0]) / na                                # 최근 구간의 평균 시각
        pred = icpt + slope * ta
        recent = (cx[i1] - cx[a0]) / na
        resid = recent - pred
        tbar = st / nr
        sxx = stt - st * st / nr
        lever = 1.0 / nr + np.where(sxx > 1e-9, (ta - tbar) ** 2 / sxx, 0.0)
    # 잡음: 지금까지(현재 제외) 1차 차분의 MAD → 수준 잡음 σ. 확장 중앙값을 직접 계산(실험당 행 수가 작아 충분히 빠름)
    d = np.diff(x, prepend=x[0])
    sigma = np.full(n, np.nan)
    for i in range(8, n):
        dd = d[1:i]
        dd = dd[~np.isnan(dd)]
        if len(dd) >= 6:
            sigma[i] = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2)
    floor = np.nanmax([0.002 * np.nanmedian(np.abs(x)) if np.isfinite(np.nanmedian(np.abs(x))) else 0.0, 1e-3])
    sigma = np.maximum(sigma, floor)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = resid / (sigma * np.sqrt(1.0 / na + lever))
    bad = (nr < min_ref) | (na < 1) | np.isnan(z) | np.isinf(z)
    return np.where(bad, 0.0, z), np.where(bad, 0.0, resid), sigma


def per_experiment(frame, fn):
    """fn(sub_frame_sorted) -> dict of arrays. 입력 행 순서와 무관하게 실험별 시간순으로 계산해 원래 인덱스로 돌려준다."""
    t_all = frame['relative_time_s'].to_numpy(float)
    out = None
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        res = fn(frame.iloc[idx])
        if out is None:
            out = {k: np.full(len(frame), np.nan) for k in res}
        for k, v in res.items():
            out[k][idx] = v
    return pd.DataFrame(out, index=frame.index)
