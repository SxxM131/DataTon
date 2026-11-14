"""3단계 = '압력 연속 상승' 이벤트(원본 1초 데이터 기준으로 판정된 것으로 추정)를 겨냥한 부호 기반 피처.

원본 공개 설명의 이상 이벤트 항목: 온도 증가 → 압력 연속 상승 → 전압 1차 급락 → 전압 크기 → 전압 2차 급락 → 온도 증가2 → 온도 크기.
우리 데이터는 5초마다 한 행이라 1초 단위의 '연속 상승'을 직접 볼 수는 없다. 대신 그 이벤트가 일어날 확률을 좌우하는
'압력이 오르는 비율'과 '연속 상승이 이어진 시간'을 크기와 무관한(부호만 쓰는) 피처로 만든다.
모두 같은 실험의 현재·과거 행만 사용하고, 길이는 초 단위다 (Test의 10/15초 간격 대응).
"""
import numpy as np
import pandas as pd

WINDOWS = (30.0, 60.0, 120.0, 300.0)


def _one(t, p, t1):
    n = len(t)
    out = {}
    d = np.diff(p, prepend=p[0])
    up = (d > 0).astype(float); up[0] = 0.0
    dn = (d < 0).astype(float)
    # 연속 상승이 이어진 시간(초): 현재 행까지 연속으로 올랐다면 그 구간의 시작부터 현재까지
    run_s = np.zeros(n); run_n = np.zeros(n)
    for i in range(1, n):
        if up[i]:
            run_s[i] = run_s[i - 1] + (t[i] - t[i - 1]); run_n[i] = run_n[i - 1] + 1
    out['s_run_s'] = run_s
    out['s_run_n'] = run_n
    cu = np.concatenate([[0.0], np.cumsum(up)]); c1 = np.arange(n + 1, dtype=float)
    cd = np.concatenate([[0.0], np.cumsum(dn)])
    for w in WINDOWS:
        lo = np.searchsorted(t, t - w, side='right')
        cnt = np.maximum(c1[1:] - c1[lo], 1.0)
        out[f's_upfrac{int(w)}'] = (cu[1:] - cu[lo]) / cnt
        out[f's_net{int(w)}'] = ((cu[1:] - cu[lo]) - (cd[1:] - cd[lo])) / cnt
    for thr in (45.0, 50.0):
        gate = np.maximum.accumulate(t1) >= thr
        k = int(thr)
        tg = t[gate][0] if gate.any() else np.inf
        out[f's_since{k}'] = np.where(gate, t - tg, -1.0)
        g_run = np.where(gate, run_s, 0.0)
        out[f's_runmax{k}'] = np.maximum.accumulate(g_run)                       # 게이트 이후 최장 연속 상승 시간
        gu = np.where(gate, up, 0.0)
        out[f's_upcnt{k}'] = np.cumsum(gu)                                        # 게이트 이후 상승 횟수
        rows = np.cumsum(gate.astype(float))
        out[f's_upfrac_since{k}'] = np.where(rows > 0, np.cumsum(gu) / np.maximum(rows, 1.0), 0.0)
        for q in (10.0, 15.0, 20.0, 30.0):                                        # 연속 상승이 q초 이상 이어진 첫 시점 이후 경과
            first = np.where(g_run >= q, t, np.inf)
            first = np.minimum.accumulate(first)
            out[f's_since_run{int(q)}_{k}'] = np.where(np.isfinite(first), t - first, -1.0)
        # 누적 위험: 최근 60초 상승 비율의 거듭제곱을 게이트 이후 시간에 대해 적분 (연속 상승 이벤트의 발생 확률 근사)
        uf = out['s_upfrac60']
        dt = np.diff(t, prepend=t[0])
        for pw in (3, 6):
            out[f's_haz{pw}_{k}'] = np.cumsum(np.where(gate, uf ** pw * dt, 0.0))
    return out


def sign_features(frame: pd.DataFrame) -> pd.DataFrame:
    t_all = frame['relative_time_s'].to_numpy(float)
    res = None
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        o = _one(t_all[idx], frame['meas_p'].to_numpy(float)[idx], frame['meas_t1'].to_numpy(float)[idx])
        if res is None:
            res = {k: np.full(len(frame), np.nan) for k in o}
        for k, v in o.items():
            res[k][idx] = v
    return pd.DataFrame(res, index=frame.index)
