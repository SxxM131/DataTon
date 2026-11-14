"""세션 D — 전이 규칙 후보 스캔.
각 실험에서 (onset 이전 eligible 행들의 최댓값, onset 행 값] 구간을 만들고, 가장 많은 실험의 구간을 관통하는 임계값 θ를 찾는다
('f >= θ가 처음 참이 되는 행 == onset'이 되는 실험 수). 방향 반대(-f)도 같이 본다.
"""
import sys; sys.path.insert(0, '.')
from lib import *


def cand_feats(pool):
    """분석용 후보 피처 (전부 인과적: 현재+과거만). pool은 실험·시간 정렬 상태."""
    g = pool.groupby('experiment_id', sort=False)
    F = pd.DataFrame(index=pool.index)
    t = pool.relative_time_s
    base = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'thermal_t', 'indu_t', 'atmp_t'] + GAS
    for c in base:
        F[c] = pool[c]
    F['t1+t2'] = pool.meas_t1 + pool.meas_t2
    F['max12'] = pool[['meas_t1', 'meas_t2']].max(1)
    F['min12'] = pool[['meas_t1', 'meas_t2']].min(1)
    F['t2-t1'] = pool.meas_t2 - pool.meas_t1
    F['t1-atmp'] = pool.meas_t1 - pool.atmp_t
    F['t2-atmp'] = pool.meas_t2 - pool.atmp_t
    F['th-t1'] = pool.thermal_t - pool.meas_t1
    F['gas_sum'] = pool[GAS].sum(1)
    v0 = g.meas_v.transform('first'); vmax = g.meas_v.cummax()
    F['v-v0'] = pool.meas_v - v0; F['v/v0'] = pool.meas_v / v0
    F['v-vmax'] = pool.meas_v - vmax; F['v/vmax'] = pool.meas_v / vmax
    for c in ['meas_t1', 'meas_t2', 'meas_p', 'thermal_t', 'indu_t']:
        F[c + '-first'] = pool[c] - g[c].transform('first')
        F[c + '_cmax'] = g[c].cummax()
        F[c + '-cmax'] = pool[c] - F[c + '_cmax']
    for c in ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'thermal_t', 'indu_t', 't1+t2']:
        s = F[c]
        for L in (1, 2, 3, 6, 12, 24, 60):
            F[f'{c}_d{L*5}'] = s - s.groupby(pool.experiment_id, sort=False).shift(L)
        F[f'{c}_dd5'] = F[f'{c}_d5'] - F[f'{c}_d5'].groupby(pool.experiment_id, sort=False).shift(1)
    for c in GAS + ['gas_sum']:
        for W in (6, 12, 24):
            F[f'{c}_m{W*5}'] = F[c].groupby(pool.experiment_id, sort=False).rolling(W, min_periods=1).mean().reset_index(level=0, drop=True)
        F[c + '_cmax'] = F[c].groupby(pool.experiment_id, sort=False).cummax()
    F['time'] = t
    t45 = t.where(pool.meas_t1 >= 45).groupby(pool.experiment_id, sort=False).cummin()
    F['since45'] = t - t45
    t50 = t.where(pool.meas_t1 >= 50).groupby(pool.experiment_id, sort=False).cummin()
    F['since50'] = t - t50
    return F


def stab(lo, hi):
    """구간 (lo, hi] 를 가장 많이 관통하는 θ와 그 수. lo<hi 인 것만 유효."""
    ok = hi > lo
    lo, hi = lo[ok], hi[ok]
    if len(lo) == 0:
        return 0, np.nan
    pts = np.unique(hi)
    cnt = [(np.sum((lo < p) & (p <= hi)), p) for p in pts]
    n, p = max(cnt)
    return n, p


def scan(pool, F, k, elig=None, groups=None, top=25, min_show=None):
    """k: 새 단계. elig: onset 이전 행 중 '규칙이 평가되는' 행 마스크 (기본: 이전 단계(실제 라벨 k-1 이하이면서 직전 단계 시작 이후) 행)."""
    on = onsets(pool, k)
    exps = on.index
    res = []
    lab = pool.label.to_numpy()
    eid = pool.experiment_id.to_numpy()
    row = pool.row.to_numpy()
    onrow = pd.Series(eid).map(on).to_numpy()
    pre = (row < onrow)
    if elig is not None:
        pre &= elig
    is_on = row == onrow
    for c in F.columns:
        x = F[c].to_numpy(dtype=float)
        hi = pd.Series(x[is_on], index=eid[is_on])
        for sgn, name in ((1, c + ' >='), (-1, c + ' <=')):
            xs = sgn * x
            lo = pd.Series(np.where(pre, xs, np.nan)).groupby(eid).max().reindex(hi.index).fillna(-np.inf)
            h = sgn * hi
            v = h.notna()
            n, p = stab(lo[v].to_numpy(), h[v].to_numpy())
            res.append((name, n, sgn * p, int(v.sum())))
    R = pd.DataFrame(res, columns=['rule', 'n_exact', 'theta', 'n_valid']).sort_values('n_exact', ascending=False)
    return R.head(top)


if __name__ == '__main__':
    pool = load()
    fp = CACHE / 'cand.pkl'
    F = cand_feats(pool); F.to_pickle(fp)
    print(F.shape)


def scan_tol(pool, F, k, elig=None, top=25, late=1, early=0, cols=None):
    """scan과 같지만 허용 오차 포함: 규칙이 onset-early ~ onset+late 행에서 처음 참이 되면 적중.
    구간: lo = (onset-early 이전 eligible 행 최댓값), hi = onset..onset+late 행의 최댓값."""
    on = onsets(pool, k)
    eid = pool.experiment_id.to_numpy(); row = pool.row.to_numpy()
    onrow = pd.Series(eid).map(on).to_numpy()
    pre = row < onrow - early
    if elig is not None:
        pre &= elig
    win = (row >= onrow) & (row <= onrow + late)
    res = []
    for c in (cols or F.columns):
        x = F[c].to_numpy(dtype=float)
        for sgn, name in ((1, c + ' >='), (-1, c + ' <=')):
            xs = sgn * x
            hi = pd.Series(np.where(win, xs, np.nan)).groupby(eid).max().reindex(on.index)
            lo = pd.Series(np.where(pre, xs, np.nan)).groupby(eid).max().reindex(on.index).fillna(-np.inf)
            v = hi.notna().to_numpy()
            n, p = stab(lo.to_numpy()[v], hi.to_numpy()[v])
            res.append((name, n, sgn * p, int(v.sum())))
    return pd.DataFrame(res, columns=['rule', 'n_hit', 'theta', 'n_valid']).sort_values('n_hit', ascending=False).head(top)
