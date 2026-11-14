"""세션 A — 공통 검증 체계(cv.py)로 테이블 모델(LightGBM) 변형을 채점하는 러너.

구조는 experiments/common/ref_gbdt.py와 같다:
  피처는 pool(원본 5초 간격) / evalp(Test처럼 다운샘플된 행)에서 각각 따로 생성 → fold 루프 → evaluate.
추가 옵션:
  --train_rows : pool(원본) | tl(Test-like 행만) | pool+tl(원본 + 긴 학습 실험의 Test-like 복사본) | pool+tl3(+ 위상 다른 다운샘플 2개)
  --weight     : none | exp(실험별 총가중치 동일) | sqrt(실험 행 수의 -1/2승) | class(클래스 균형) | exp+class
  --feat       : sensor5(센서 5개 + 이미지) | all(+ 가스·환경·메타)
  --cfg        : feats.DEFAULT 덮어쓰기 (JSON)
모든 증강 행은 해당 fold의 학습 실험에서만 만든다.

예) OMP_NUM_THREADS=6 python experiments/tab/run.py --name tab_x --feat all --train_rows pool+tl --n_repeats 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common')]
import cv  # noqa: E402
import feats  # noqa: E402

DATA = Path.home() / 'data/battery_data_light'
CACHE = Path.home() / 'jaehyeok/cache'
REF = HERE.parent / 'results/oof/ref_gbdt_color_extra.pkl'

LGB_DEFAULT = dict(objective='multiclass', learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=6, verbose=-1,
                   deterministic=True, force_row_wise=True)


def jitter_downsample(frame: pd.DataFrame, off: float, cap: int = cv.CAP) -> pd.DataFrame:
    """Test 다운샘플 규칙의 위상만 바꾼 변형 (floor(linspace + off)). 첫 행은 항상 유지. 긴 실험만 반환."""
    keep = []
    t = frame['relative_time_s'].to_numpy()
    for _, idx in frame.groupby('experiment_id', sort=True).indices.items():
        if len(idx) <= cap:
            continue
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        k = np.minimum(np.floor(np.linspace(0, len(idx) - 1, cap) + off).astype(int), len(idx) - 1)
        k[0] = 0
        keep.append(idx[np.unique(k)])
    return frame.iloc[np.concatenate(keep)]


def sample_weight(eids: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    w = np.ones(len(y))
    if 'exp' in mode or mode == 'sqrt':
        n = pd.Series(eids).map(pd.Series(eids).value_counts()).to_numpy(dtype=float)
        w *= (1.0 / n) if 'exp' in mode else (1.0 / np.sqrt(n))
    if 'class' in mode:
        cls = pd.Series(y)
        tot = pd.Series(w).groupby(cls.values).sum()
        w *= (tot.mean() / cls.map(tot)).to_numpy()
    return w * len(w) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--feat', default='all', choices=['sensor5', 'all'])
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--train_rows', default='pool')
    ap.add_argument('--weight', default='none')
    ap.add_argument('--lgb', default='{}')
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--n_repeats', type=int, default=1)
    ap.add_argument('--drop', default='', help='쉼표 구분 접두어/이름: 해당 피처 제외')
    ap.add_argument('--no_official', action='store_true')
    args = ap.parse_args()
    cfg = {'extra': args.feat == 'all', **json.loads(args.cfg)}
    params = {**LGB_DEFAULT, **json.loads(args.lgb)}
    t0 = time.time()

    pool = cv.load_pool(DATA)
    evalp = cv.to_test_like(pool)
    color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
    et = cv.exp_table(pool)
    long_exps = set(et.index[et.long])

    def F(fr):
        return feats.build(fr, color, cfg)
    Xp, Xe = F(pool), F(evalp)
    tb = time.time()
    F(evalp[evalp.experiment_id.isin(sorted(evalp.experiment_id.unique())[:11])])
    t_build = time.time() - tb
    if args.drop:
        dr = [d for d in args.drop.split(',') if d]
        keep = [c for c in Xp.columns if not any(c == d or c.startswith(d) or (d.startswith('*') and d[1:] in c) for d in dr)]
        Xp, Xe = Xp[keep], Xe[keep]
    cols = list(Xp.columns)
    # 증강 세트: (피처, 라벨, 실험ID)
    aug = []
    if args.train_rows.startswith('pool+tl'):
        tl = evalp[evalp.experiment_id.isin(long_exps)]
        aug.append((Xe.loc[tl.index], tl.label.to_numpy(int), tl.experiment_id.to_numpy() + '#tl'))
    if args.train_rows == 'pool+tl3':
        for off in (0.34, 0.67):
            j = jitter_downsample(pool, off)
            aug.append((F(j)[cols], j.label.to_numpy(int), j.experiment_id.to_numpy() + f'#j{off}'))
    print(f'[{args.name}] 피처 {len(cols)}개, pool {len(pool)} / evalp {len(evalp)}, 증강 {sum(len(a[1]) for a in aug)}행, '
          f'준비 {time.time()-t0:.1f}s, 11실험 피처생성 {t_build*1000:.0f}ms', flush=True)

    def train_set(train_exps):
        tr = set(train_exps)
        if args.train_rows == 'tl':
            m = evalp.experiment_id.isin(tr).to_numpy()
            Xs, ys, es = [Xe[m]], [evalp.label.to_numpy(int)[m]], [evalp.experiment_id.to_numpy()[m]]
        else:
            m = pool.experiment_id.isin(tr).to_numpy()
            Xs, ys, es = [Xp[m]], [pool.label.to_numpy(int)[m]], [pool.experiment_id.to_numpy()[m]]
        for Xa, ya, ea in aug:
            m = np.isin(np.array([e.split('#')[0] for e in ea]), list(tr))
            Xs.append(Xa[m]); ys.append(ya[m]); es.append(ea[m])
        return pd.concat(Xs), np.concatenate(ys), np.concatenate(es)

    def run(f):
        X, y, e = train_set(f['train_exps'])
        w = sample_weight(e, y, args.weight)
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        P = 0
        for s in range(args.seeds):
            m = lgb.LGBMClassifier(**params, random_state=42 + s).fit(X, y, sample_weight=w)
            P = P + m.predict_proba(Xe[va]) / args.seeds
        return evalp.index[va], P, m

    oof = pd.DataFrame(index=evalp.index)
    imp = 0
    for f in cv.make_folds(pool, n_repeats=args.n_repeats):
        idx, P, m = run(f)
        r = f['rep']
        oof.loc[idx, f'pred_r{r}'] = np.argmax(P, 1) + 1
        for k in range(6):
            oof.loc[idx, f'p{k+1}_r{r}'] = P[:, k]
        imp = imp + pd.Series(m.booster_.feature_importance('gain'), index=cols)
    for c in [c for c in oof.columns if c.startswith('pred_r')]:
        oof[c] = oof[c].astype(int)
    res = cv.evaluate(evalp, oof)
    if not args.no_official:
        idx, P, _ = run(cv.official_split(pool))
        res['official_val'] = cv.macro_f1(evalp.loc[idx, 'label'], np.argmax(P, 1) + 1)
    ref = pd.read_pickle(REF)
    cmp = cv.compare(evalp, oof, ref)
    # 긴 실험 중 이상 실험(E_e5bc…: 5단계가 ~3000초 지속) 제외 점수 — 진단용
    res['cmp_ref'] = cmp
    res['n_feat'] = len(cols)
    res['build_ms_11exp'] = t_build * 1000
    cv.save_result(args.name, res, oof, extra={'args': vars(args), 'cfg': cfg, 'lgb': params})
    print(cv.summary_line(args.name, res))
    print(f"  test11 {res['test11_mean']:.3f}±{res['test11_sd']:.3f} | vs ref diff {cmp['diff']:+.4f} "
          f"CI90 [{cmp['ci90'][0]:+.3f},{cmp['ci90'][1]:+.3f}] p_better {cmp['p_a_better']:.2f} | {time.time()-t0:.0f}s", flush=True)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(HERE / 'logs' / f'{args.name}_imp.csv')
    print('  top feats:', ', '.join(f'{k}:{v:.3f}' for k, v in imp.head(15).items()))


if __name__ == '__main__':
    main()
