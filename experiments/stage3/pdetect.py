"""3단계 단서: 압력(meas_p) 상승 시작점을 실험별 잡음 대비로 잡는 인과적 검출기.

배경: 3단계 시작 전후로 압력이 '그 실험의 추세 대비' 오르기 시작한다 (사건 연구: 52개 실험 중 83%).
실험마다 압력 수준·잡음이 달라 고정 임계값으로는 안 잡히므로, 과거 관측만으로 잡음(σ)과 기준선을 추정해 z-점수로 본다.
모든 계산은 같은 실험의 현재·과거 행만 사용 (초 단위 윈도우).
"""
import numpy as np
import pandas as pd


def _one(t, p, t1, a_s=10.0, gap_s=15.0, ref_s=120.0, min_sigma=0.02):
    """한 실험(시간순). 반환: z (최근 평균 − 기준선)/σ, sigma, cusum."""
    n = len(t)
    z = np.zeros(n); sig = np.full(n, np.nan); cus = np.zeros(n)
    d = np.diff(p, prepend=p[0])
    absdev = []
    s = 0.0
    for i in range(n):
        # 잡음: 지금까지의 1차 차분의 MAD (확장 윈도우, 현재 행 제외) → 수준 잡음 σ = MAD*1.4826/√2
        if i >= 8:
            dd = d[1:i]
            m = np.median(dd)
            sigma = max(1.4826 * np.median(np.abs(dd - m)) / np.sqrt(2), min_sigma)
        else:
            sigma = np.nan
        sig[i] = sigma
        lo_a = np.searchsorted(t, t[i] - a_s, side='right')            # 최근 a_s초 (현재 포함)
        hi_r = np.searchsorted(t, t[i] - gap_s, side='right')          # 기준 구간 끝 (t-gap 이하)
        lo_r = np.searchsorted(t, t[i] - gap_s - ref_s, side='right')
        if np.isnan(sigma) or hi_r - lo_r < 4:
            continue
        recent = p[lo_a:i + 1]
        ref = p[lo_r:hi_r]
        se = sigma * np.sqrt(1.0 / len(recent) + 1.57 / len(ref))      # 중앙값의 분산 ≈ 1.57σ²/n
        z[i] = (recent.mean() - np.median(ref)) / se
        # CUSUM: 표준화된 증가분에서 여유 0.5σ를 뺀 누적 (음수면 0으로)
        s = max(0.0, s + (d[i] / (sigma * np.sqrt(2))) - 0.35)
        cus[i] = s
    return z, sig, cus


def detect(frame: pd.DataFrame, **kw) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=['pz', 'psig', 'pcus'], dtype=float)
    t_all = frame['relative_time_s'].to_numpy(float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        z, s, c = _one(t_all[idx], frame['meas_p'].to_numpy(float)[idx], frame['meas_t1'].to_numpy(float)[idx], **kw)
        out.iloc[idx, 0] = z; out.iloc[idx, 1] = s; out.iloc[idx, 2] = c
    return out
