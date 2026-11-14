"""[라운드 2-2] 인과적 프레임 변화(이벤트) 피처 — 소수 정예 12개.

근거(onset_check): bat2 계열은 벤팅 구름이 라벨 4보다 1행, 전압 급락보다 2행 먼저 화면을 덮는다. 소형 셀은 변화 없음.
정의 (보이는 행만 사용; 직전 관측 = 시간상 바로 앞 행, 5~15초 전):
  ev_dprev      : mean|pt32 - 직전 pt32|          ev_fprev : |Δ|>0.15 인 픽셀 비율 (변화 면적)
  ev_d15        : t-15초 이하 마지막 관측과의 mean|Δ|
  ev_r_runmax   : ev_dprev / (t1>=45 도달 이후 과거 ev_dprev 최댓값)   ev_z_runmed : ev_dprev / (같은 구간 과거 중앙값)
  ev_dprev_lag1 : 직전 행의 ev_dprev                ev_fprev_lag1
  ev_dmax30     : 최근 30초 ev_dprev 최댓값
  ev_since_first: t1>=45 이후 '큰 변화'(ev_fprev>0.3, 고정 임계) 첫 발생 이후 경과 초 (없으면 -1)
  ev_since_last : 마지막 큰 변화 이후 경과 초 (없으면 -1)      ev_n : 큰 변화 누적 횟수
  ev_now        : 현재 행이 큰 변화인가
"""
import numpy as np
import pandas as pd
import imglib as L

BIG = 0.3


def build(pool, P, mask):
    n = len(pool)
    t = pool.relative_time_s.to_numpy(float); t1 = pool.meas_t1.to_numpy(float)
    cols = ['ev_dprev', 'ev_fprev', 'ev_d15', 'ev_r_runmax', 'ev_z_runmed', 'ev_dprev_lag1', 'ev_fprev_lag1', 'ev_dmax30',
            'ev_since_first', 'ev_since_last', 'ev_n', 'ev_now']
    out = np.full((n, len(cols)), np.nan, dtype=np.float32)
    for _, idx in pool.groupby('experiment_id', sort=False).indices.items():
        idx = idx[mask[idx]]
        if not len(idx):
            continue
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        tt = t[idx]; m = len(idx)
        dif = np.abs(P[idx[1:]] - P[idx[:-1]])
        d = np.r_[0, dif.mean(1)]; f = np.r_[0, (dif > 0.15).mean(1)]
        j15 = np.maximum(np.searchsorted(tt, tt - 15.0, side='right') - 1, 0)
        d15 = np.abs(P[idx] - P[idx[j15]]).mean(1)
        hot = np.maximum.accumulate(np.nan_to_num(t1[idx], nan=-1) >= 45)            # 지금까지 t1>=45 도달 여부 (인과적)
        dg = np.where(hot, d, np.nan); dg[np.argmax(hot)] = np.nan if hot.any() else np.nan   # 도달 행 자신의 차이는 제외
        s = pd.Series(dg)
        runmax = s.expanding().max().shift(1).to_numpy(); runmed = s.expanding().median().shift(1).to_numpy()
        big = hot & (f > BIG)
        st = np.searchsorted(tt, tt - 30.0, side='right')
        dmax30 = np.array([d[a:b + 1].max() for a, b in zip(st, range(m))])
        tb = np.where(big, tt, np.nan)
        first = pd.Series(tb).expanding().min().to_numpy(); last = pd.Series(tb).ffill().to_numpy()
        out[idx] = np.c_[d, f, d15, d / (runmax + 1e-3), d / (runmed + 1e-3), np.r_[0, d[:-1]], np.r_[0, f[:-1]], dmax30,
                         np.where(np.isnan(first), -1, tt - first), np.where(np.isnan(last), -1, tt - last), np.cumsum(big), big]
    return pd.DataFrame(out[mask], index=pool.index[mask], columns=cols)


def get(pool, evalp):
    f = L.ICACHE / 'evfeat.pkl'
    if f.exists():
        return pd.read_pickle(f)
    pt = L.load_images(pool, 'pt', 128, mmap=False)
    P = pt.reshape(len(pt), 32, 4, 32, 4).mean((2, 4), dtype=np.float32).reshape(len(pt), -1) / 255.0
    is_eval = np.zeros(len(pool), bool); is_eval[pool.index.get_indexer(evalp.index)] = True
    out = build(pool, P, np.ones(len(pool), bool)), build(pool, P, is_eval)
    pd.to_pickle(out, f)
    return out
