"""열화상의 상대 눈금(프레임마다 자동 조정된 팔레트)을 절대 온도로 바꿀 수 있는가, 바꾸면 도움이 되는가.

가정: 프레임의 가장 차가운 색 = 주위 온도(atmp_t), 가장 뜨거운 색 = 열화상 카메라가 기록한 온도(thermal_t).
      → 화소 온도 T = atmp_t + s × (thermal_t − atmp_t),  s = 팔레트 역변환 값(0~1, cache_images.pseudo_temp).
(1) 이 가정이 맞는지: 변환한 화면 온도가 표면 온도 센서(meas_t1)를 thermal_t 단독이나 상대값 s보다 잘 따라가는가 (실험 안 상관).
(2) 도움이 되는지: 변환한 절대 온도 피처(평균·중앙부·분위·임계 온도 이상 면적, 그 변화량)를 테이블 모델에 더해 공통 검증(3회) + 규칙 체인.
사용: python experiments/img/abs_temp.py
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common'), str(HERE)]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
ids = np.load(C / 'img' / 'pool_ids.npy'); assert (ids == pool.sample_id.to_numpy().astype(str)).all()
PT = np.load(C / 'img' / 'pool_pt128.npy', mmap_mode='r')                      # (N, 128, 128) uint8, 상대 온도 0~255
TH = (40, 50, 60, 80, 100, 150)


def frame_feats():
    f = C / 'img' / 'abs_temp_feats.pkl'
    if f.exists():
        return pd.read_pickle(f)
    at = pool.atmp_t.to_numpy(float); d = np.clip(pool.thermal_t.to_numpy(float) - at, 0, None); n = len(pool); out = {}
    sm, sc, s50, s90 = (np.zeros(n) for _ in range(4)); fr = np.zeros((n, len(TH)))
    for i in range(0, n, 2000):
        s = np.asarray(PT[i:i + 2000], dtype=np.float32) / 255.0; j = slice(i, i + len(s)); flat = s.reshape(len(s), -1)
        sm[j] = flat.mean(1); sc[j] = s[:, 32:96, 32:96].reshape(len(s), -1).mean(1); s50[j] = np.percentile(flat, 50, axis=1); s90[j] = np.percentile(flat, 90, axis=1)
        for k, th in enumerate(TH):
            need = (th - at[j]) / np.maximum(d[j], 1e-6)                             # 이 온도가 되려면 필요한 s
            fr[j, k] = (flat >= need[:, None]).mean(1) * (d[j] > 0)
    out = {'s_mean': sm, 's_center': sc, 'abs_mean': at + sm * d, 'abs_center': at + sc * d, 'abs_p50': at + s50 * d, 'abs_p90': at + s90 * d, 'abs_range': d}
    for k, th in enumerate(TH):
        out[f'abs_area{th}'] = fr[:, k]
    df = pd.DataFrame(out, index=pool.index); df.to_pickle(f)
    return df


F = frame_feats()
# ---- (1) 가정 점검: 실험 안에서 meas_t1과의 상관 (1~3단계 행, t1이 30℃ 넘게 변한 실험만) ----
rows = []
for ex, g in pool.groupby('experiment_id'):
    g = g[g.label <= 3]
    if len(g) < 20 or g.meas_t1.max() - g.meas_t1.min() < 30:
        continue
    f = F.loc[g.index]; t1 = g.meas_t1
    rows.append({'thermal_t': t1.corr(g.thermal_t), '상대값 평균 s': t1.corr(f.s_mean), '상대값 중앙부 s': t1.corr(f.s_center), '변환 평균 온도': t1.corr(f.abs_mean), '변환 중앙부 온도': t1.corr(f.abs_center), '변환 p90': t1.corr(f.abs_p90),
                 'err_thermal': float((g.thermal_t - t1).abs().median()), 'err_abs_center': float((f.abs_center - t1).abs().median()), 'err_abs_p90': float((f.abs_p90 - t1).abs().median())})
Rw = pd.DataFrame(rows)
print(f'(1) 실험 {len(Rw)}개, 1~3단계 행에서 표면 온도(meas_t1)와의 상관 — 중앙값 [하위 25%]')
for c in ('thermal_t', '상대값 평균 s', '상대값 중앙부 s', '변환 평균 온도', '변환 중앙부 온도', '변환 p90'):
    print(f'    {c:14s} {Rw[c].median():.3f} [{Rw[c].quantile(0.25):.3f}]')
print(f"    meas_t1과의 절대 차이 중앙값(℃): thermal_t {Rw.err_thermal.median():.1f} | 변환 중앙부 {Rw.err_abs_center.median():.1f} | 변환 p90 {Rw.err_abs_p90.median():.1f}")
# 프레임 사이 매끄러움: 변환 온도는 물리량이면 천천히 변해야 한다 (1~3단계, 직전 행 대비 변화의 중앙 절대값 / 전체 변화폭)
sm = []
for ex, g in pool[pool.label <= 3].groupby('experiment_id'):
    if len(g) < 20:
        continue
    f = F.loc[g.sort_values('relative_time_s').index]
    sm.append({c: float(f[c].diff().abs().median() / (f[c].max() - f[c].min() + 1e-9)) for c in ('s_center', 'abs_center', 'abs_range')})
print('    프레임 간 변화(변화폭 대비, 작을수록 매끄러움) 중앙값:', pd.DataFrame(sm).median().round(4).to_dict(), flush=True)

# ---- (2) 테이블 모델에 더해 공통 검증 ----
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
RAW = [c for c in F.columns if c.startswith('abs_')]


def with_lags(frame):
    """절대 온도 피처 + 15·60초 전 대비 변화 (그 frame에 보이는 과거 행만)."""
    f = F.loc[frame.index, RAW].copy(); codes, t_all, order = M._sort_order(frame); g, t = codes[order], t_all[order]
    for c in ('abs_mean', 'abs_center', 'abs_area50', 'abs_area80'):
        x = f[c].to_numpy()[order]
        for w in (15, 60):
            v = np.full(len(f), np.nan); v[order] = x - M._lag(g, t, x, w); f[f'{c}_d{w}s'] = v
    return f


Ap, Ae = with_lags(pool), with_lags(evalp)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
cur = pd.DataFrame({f'pred_r{r}': M.apply_rules(A[f'pred_r{r}'].to_numpy(int), pb[r], evalp) for r in range(3)}, index=evalp.index)
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])
mf = lambda d, m=None: float(np.mean([cv.macro_f1(ye if m is None else ye[m], d[k].to_numpy() if m is None else d[k].to_numpy()[m]) for k in d.columns]))  # noqa: E731
print(f'\n(2) 현재: 규칙 체인 후 {mf(cur):.4f} | Test 구성 {mf(cur, TEST):.4f}', flush=True)
for name, cols in ((f'절대 온도 피처 {len(RAW)}개 (값만)', RAW), (f'절대 온도 피처 + 변화량 ({Ap.shape[1]}개)', list(Ap.columns))):
    raw, fin = pd.DataFrame(index=evalp.index), pd.DataFrame(index=evalp.index)
    for r in range(3):
        p = np.zeros((len(evalp), 6))
        for f in cv.make_folds(pool, n_repeats=3):
            if f['rep'] != r:
                continue
            tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
            m = lgb.LGBMClassifier(**{**M.LGB_PARAMS, 'n_jobs': 6}).fit(pd.concat([Xp[tr], Ap.loc[pool.index[tr], cols]], axis=1), y[tr])
            p[va] = m.predict_proba(pd.concat([Xe[va], Ae.loc[evalp.index[va], cols]], axis=1))
        raw[f'pred_r{r}'] = p.argmax(1) + 1; fin[f'pred_r{r}'] = M.apply_rules(raw[f'pred_r{r}'].to_numpy(), pb[r], evalp)
    c = cv.compare(evalp, fin, cur, n_boot=2000); pc = np.mean([__import__('sklearn.metrics').metrics.f1_score(ye, fin[k], labels=range(1, 7), average=None) for k in fin.columns], axis=0)
    print(f"    {name}: 테이블 단독 {mf(raw):.4f} | 규칙 체인 후 {mf(fin):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(fin, TEST):.4f} ({mf(fin, TEST) - mf(cur, TEST):+.4f}) | 단계별 {np.round(pc, 3).tolist()}", flush=True)
