"""손으로 만든 '이미지 변화' 피처 (인과적). 의사온도 맵을 32×32로 줄여 참조 프레임(첫 프레임, 30초 전, 직전)과 비교.

팔레트가 프레임마다 자동 스케일되는 것으로 보여(초기 프레임은 노이즈가 팔레트 전 범위로 늘어남) 절대 온도 대신
'장면이 얼마나/언제 급변했는가'를 잡는다: 벤팅·화염 때 프레임이 통째로 바뀜.
pool 문맥(학습 행) / evalp 문맥(평가 행)에서 각각 계산.
"""
import numpy as np
import pandas as pd
import imglib as L

COLS = None


def _corr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-6)


def build(pool, P, mask):
    """P: (N, 1024) float32 의사온도(0~1). mask: 보이는 행."""
    ctx = L.context_index(pool, mask, lags=(30, 60))
    rows = np.where(mask)[0]
    t = pool.relative_time_s.to_numpy(dtype=float)
    f = {}
    cur = P[rows]
    img = cur.reshape(-1, 32, 32)
    f['hf_noise'] = (np.abs(np.diff(img, axis=1)).mean((1, 2)) + np.abs(np.diff(img, axis=2)).mean((1, 2))) / 2   # 고주파 에너지(자동 스케일 범위의 대리값)
    f['pt_mean32'] = cur.mean(1); f['pt_std32'] = cur.std(1)
    gy, gx = np.mgrid[0:32, 0:32] / 31.0
    hot = np.clip(img - 0.6, 0, None) + 1e-6
    f['hot_cy'] = (hot * gy).sum((1, 2)) / hot.sum((1, 2)); f['hot_cx'] = np.abs((hot * gx).sum((1, 2)) / hot.sum((1, 2)) - .5)
    f['border_mean'] = np.concatenate([img[:, :3].reshape(len(img), -1), img[:, -3:].reshape(len(img), -1),
                                       img[:, :, :3].reshape(len(img), -1), img[:, :, -3:].reshape(len(img), -1)], 1).mean(1)
    f['center_mean'] = img[:, 10:22, 10:22].mean((1, 2))
    for k in ('first', 'lag30', 'lag60', 'prev'):
        ref = P[ctx[k][rows]]
        f[f'd_{k}_abs'] = np.abs(cur - ref).mean(1)
        f[f'd_{k}_sgn'] = (cur - ref).mean(1)
        f[f'corr_{k}'] = _corr(cur, ref)
    dt = np.maximum(t[rows] - t[ctx['prev'][rows]], 5.0)
    f['d_prev_rate'] = f['d_prev_abs'] / dt * 10
    df = pd.DataFrame(f, index=pool.index[rows])
    # 누적(현재까지) 최대 변화·최소 상관, 그 시점 이후 경과 시간
    df['_t'] = t[rows]; df['_e'] = pool.experiment_id.to_numpy()[rows]
    df = df.sort_values(['_e', '_t'], kind='mergesort')
    g = df.groupby('_e', sort=False)
    for c, fn in [('d_lag30_abs', 'cummax'), ('corr_lag30', 'cummin'), ('corr_first', 'cummin'), ('d_first_abs', 'cummax'), ('hf_noise', 'cummin')]:
        df[f'{c}_{fn}'] = getattr(g[c], fn)()
    is_new = df['d_lag30_abs'] >= df['d_lag30_abs_cummax'] - 1e-9
    df['t_since_maxchange'] = df['_t'] - df['_t'].where(is_new).groupby(df['_e'], sort=False).ffill()
    low = df['corr_lag30'] < 0.5                                   # 고정 임계값: 30초 전과 장면 상관이 0.5 미만 = 급변
    df['n_abrupt_sofar'] = low.groupby(df['_e'], sort=False).cumsum()
    first_abrupt = df['_t'].where(low).groupby(df['_e'], sort=False).transform(lambda s: s.ffill().cummin())
    df['t_since_first_abrupt'] = (df['_t'] - first_abrupt).fillna(-1)
    return df.drop(columns=['_t', '_e']).reindex(pool.index[rows]).astype(np.float32)


def get(pool, evalp):
    f = L.ICACHE / 'handfeat.pkl'
    if f.exists():
        return pd.read_pickle(f)
    pt = L.load_images(pool, 'pt', 128, mmap=False)
    P = pt.reshape(len(pt), 32, 4, 32, 4).mean((2, 4), dtype=np.float32).reshape(len(pt), -1) / 255.0
    is_eval = np.zeros(len(pool), bool); is_eval[pool.index.get_indexer(evalp.index)] = True
    out = build(pool, P, np.ones(len(pool), bool)), build(pool, P, is_eval)
    pd.to_pickle(out, f)
    return out


if __name__ == '__main__':
    pool, evalp = L.load_all()
    a, b = get(pool, evalp)
    print(a.shape, b.shape); print(a.describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
    print(a.join(pool.label).groupby('label').mean().T.round(3).to_string())
