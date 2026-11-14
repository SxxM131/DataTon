"""세션 D — 규칙 피처 평가 러너 (공통 검증 cv.py 사용).

  --mode rule_only : 학습 없는 순수 규칙 예측 (3단계 임계 t3만 fold의 학습 실험에서 고름)
  --mode rules_lgb : 규칙 피처만으로 LightGBM
  --mode tab+rules : experiments/tab/feats.build 피처 + 규칙 피처로 LightGBM (tab 폴더는 수정하지 않고 import만)
  --mode tab       : 같은 설정의 기준선 (tab 피처만) — 같은 n_jobs/시드에서 직접 비교하기 위한 것
피처는 pool(원본 5초) / evalp(Test식 다운샘플 행)에서 각각 따로 만든다. 평가 행 피처는 evalp 행만 본다.

예) OMP_NUM_THREADS=4 python experiments/rules/run_rules.py --name rules_tab_all --mode tab+rules --n_repeats 1
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
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab')]
import cv  # noqa: E402
import feats  # noqa: E402
import rules  # noqa: E402

DATA = Path.home() / 'data/battery_data_light'
CACHE = Path.home() / 'jaehyeok/cache'
OOF = HERE.parent / 'results/oof'
LGB_DEFAULT = dict(objective='multiclass', learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=4, verbose=-1,
                   deterministic=True, force_row_wise=True)
T3_GRID = (50, 52, 55, 58, 60, 62, 65, 68, 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mode', default='tab+rules', choices=['rule_only', 'rules_lgb', 'tab+rules', 'tab'])
    ap.add_argument('--cfg', default='{}', help='tab feats cfg (extra=True 기본)')
    ap.add_argument('--lgb', default='{}')
    ap.add_argument('--n_repeats', type=int, default=1)
    ap.add_argument('--drop_rules', default='', help='쉼표 구분: 제외할 규칙 피처 접두어')
    ap.add_argument('--compare', default='tab_all_pool,ref_gbdt_color_extra')
    args = ap.parse_args()
    t0 = time.time()
    pool = cv.load_pool(DATA)
    evalp = cv.to_test_like(pool)
    folds = cv.make_folds(pool, n_repeats=args.n_repeats)
    oof = pd.DataFrame(index=evalp.index)
    extra = {'args': vars(args)}

    if args.mode == 'rule_only':
        # 학습 없음. 3단계 근사 임계 t3만 학습 실험(원본 행)에서 Macro F1이 가장 높은 값으로 고른다.
        pred_pool = {t3: rules.rule_predict(pool, t3=t3) for t3 in T3_GRID}
        pred_eval = {t3: rules.rule_predict(evalp, t3=t3) for t3 in T3_GRID}
        chosen = []
        for f in folds:
            tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
            best = max(T3_GRID, key=lambda t3: cv.macro_f1(pool.label[tr], pred_pool[t3][tr]))
            chosen.append(best)
            va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
            idx = evalp.index[va]
            p = pred_eval[best].loc[idx].to_numpy()
            oof.loc[idx, f"pred_r{f['rep']}"] = p
            for k in range(6):
                oof.loc[idx, f"p{k+1}_r{f['rep']}"] = (p == k + 1).astype(float)
        extra['t3_chosen'] = chosen
    else:
        cfg = {'extra': True, **json.loads(args.cfg)}
        params = {**LGB_DEFAULT, **json.loads(args.lgb)}
        extra.update(cfg=cfg, lgb=params)
        parts_p, parts_e = [], []
        if args.mode in ('tab+rules', 'tab'):
            color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
            parts_p.append(feats.build(pool, color, cfg)); parts_e.append(feats.build(evalp, color, cfg))
        if args.mode != 'tab':
            Rp, Re = rules.rule_features(pool), rules.rule_features(evalp)
            dr = [d for d in args.drop_rules.split(',') if d]
            keep = [c for c in Rp.columns if not any(c.startswith(d) for d in dr)]
            parts_p.append(Rp[keep]); parts_e.append(Re[keep])
        if args.mode == 'rules_lgb':   # 규칙 피처만 쓸 때는 메타 4개를 같이 준다 (규칙이 배터리 종류에 따라 다르므로)
            meta = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
            parts_p.append(pool[meta]); parts_e.append(evalp[meta])
        Xp, Xe = pd.concat(parts_p, axis=1), pd.concat(parts_e, axis=1)
        cols = list(Xp.columns)
        print(f'[{args.name}] mode {args.mode}, 피처 {len(cols)}개, 준비 {time.time()-t0:.1f}s', flush=True)
        y = pool.label.to_numpy(int)
        imp = 0
        for f in folds:
            tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
            va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
            m = lgb.LGBMClassifier(**params, random_state=42).fit(Xp[tr], y[tr])
            P = m.predict_proba(Xe[va])
            idx, r = evalp.index[va], f['rep']
            oof.loc[idx, f'pred_r{r}'] = np.argmax(P, 1) + 1
            for k in range(6):
                oof.loc[idx, f'p{k+1}_r{r}'] = P[:, k]
            imp = imp + pd.Series(m.booster_.feature_importance('gain'), index=cols)
            print(f"  rep {r} fold {f['fold']} done {time.time()-t0:.0f}s", flush=True)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        imp.to_csv(HERE / 'logs' / f'{args.name}_imp.csv')
        extra['n_feat'] = len(cols)
        print('  top feats:', ', '.join(f'{k}:{v:.3f}' for k, v in imp.head(20).items()))
        print('  rule feats rank:', ', '.join(f'{k}:{list(imp.index).index(k)+1}' for k in imp.index if k.startswith('r_'))[:1500])
    for c in [c for c in oof.columns if c.startswith('pred_r')]:
        oof[c] = oof[c].astype(int)
    res = cv.evaluate(evalp, oof)
    for other in [o for o in args.compare.split(',') if o]:
        fp = OOF / f'{other}.pkl'
        if fp.exists():
            res[f'cmp_{other}'] = cv.compare(evalp, oof, pd.read_pickle(fp))
    cv.save_result(args.name, res, oof, extra=extra)
    print(cv.summary_line(args.name, res))
    for k, c in res.items():
        if k.startswith('cmp_'):
            print(f"  vs {k[4:]:28s} diff {c['diff']:+.4f} CI90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p_better {c['p_a_better']:.2f} (repeats {c['repeats']})")
    print(f'  총 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
