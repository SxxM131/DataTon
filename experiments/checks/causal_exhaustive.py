"""미래 정보 미사용 촘촘한 점검: 실험을 여러 시점에서 잘라 넣었을 때, 남은 행의 예측이 전체를 넣었을 때와 같은가.

(1) Test 11개 실험: 실험마다 약 30개 지점(균등 + 무작위). 테이블 피처 → 두 LightGBM → 융합 → 규칙까지 다시 계산한다.
    CNN 확률은 행과 직전 행 이미지만 쓰므로 한 번 계산해 재사용하고, 그 가정은 (2)에서 직접 확인한다.
(2) CNN 포함 전체 파이프라인: Test에서 무작위 8개 (실험, 지점)을 잘라 CNN까지 다시 계산해 비교.
(3) 학습 실험 27개(구성 9종 × 3): 실험마다 15개 지점, 피처·모델·규칙 (CNN은 균등 확률로 대체 — 규칙·피처의 인과성만 본다).
"""
import os, sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
import joblib, numpy as np, pandas as pd, torch

REPO = Path(sys.argv[1]).resolve(); FINAL = Path(sys.argv[2]) if len(sys.argv) > 2 else None
sys.path[:0] = [str(REPO)]
import model as M  # noqa: E402

D = Path.home() / 'data' / 'battery_data_light'; W = REPO / 'weights'
bundle = joblib.load(W / 'tab_lgbm.joblib'); rng = np.random.default_rng(0)


def tab_part(frame, color):
    if bundle.get('feature_mode', 'full') == 'img10':
        X = M.build_features_img10(frame, color, bundle['non_img'])[bundle['columns']]
    else:
        X = M.build_features(frame, color)[bundle['columns']]
    p_tab = bundle['model'].predict_proba(X)
    X23 = pd.concat([X, M.pressure_features(frame, bundle['range_table'])], axis=1)[bundle['columns23']]
    return p_tab, bundle['model23'].predict_proba(X23)


def final(frame, color, p_cnn):
    p_tab, p23 = tab_part(frame, color)
    return M.apply_rules(M.to_label(M.fuse(p_tab, p_cnn, bundle['fuse_w'])), p23, frame)


def cuts_for(n, k):
    c = set(np.linspace(3, n - 1, min(k, n - 3)).astype(int).tolist()) | set(rng.integers(3, n, 8).tolist())
    return sorted(c)


def sweep(frame, color, p_cnn, k, name):
    frame = frame.reset_index(drop=True); full = final(frame, color, p_cnn)
    bad = tot = runs = 0
    for e, g in frame.groupby('experiment_id', sort=False):
        g = g.sort_values('relative_time_s'); idx = g.index.to_numpy()
        for c in cuts_for(len(idx), k):
            keep = np.sort(idx[:c + 1]); sub = frame.loc[keep].sample(frac=1, random_state=int(c))      # 잘라서 행 순서도 섞는다
            out = pd.Series(final(sub, color, p_cnn[sub.index.to_numpy()]), index=sub.index)
            bad += int((out.reindex(keep).to_numpy() != full[keep]).sum()); tot += len(keep); runs += 1
    print(f'{name}: 자른 입력 {runs}개, 비교한 행 {tot:,}개, 전체 입력 때와 다른 예측 {bad}행', flush=True)
    return full


# (1) Test
te = pd.read_csv(D / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
paths = M.resolve_image_paths(te, [D / 'test', D]); color = M.extract_image_features(te, paths)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M.set_torch_deterministic()
cnn = M.load_cnn(W / 'cnn_pairdiff.pt', dev); images = M.decode_images(paths); p_cnn = M.cnn_probs(te, images, cnn, dev)
full = sweep(te, color, p_cnn, 30, '(1) Test 11개 실험')
if FINAL is not None:
    sub = pd.read_csv(FINAL, dtype={'sample_id': 'string'}).set_index('sample_id').label.reindex(te.sample_id).to_numpy()
    print(f'    전체 입력 예측 = {FINAL.name}:', bool((full == sub).all()), f'(다른 행 {int((full != sub).sum())})')
# (2) CNN까지 다시
bad = tot = 0
exps = te.experiment_id.unique()
for _ in range(8):
    e = rng.choice(exps); g = te[te.experiment_id == e].sort_values('relative_time_s'); c = int(rng.integers(5, len(g)))
    keep = g.index.to_numpy()[:c + 1]; subf = te.loc[keep].sample(frac=1, random_state=c)
    pc = M.cnn_probs(subf, images[subf.index.to_numpy()], cnn, dev)
    bad += int((np.abs(pc - p_cnn[subf.index.to_numpy()]) > 0).any(1).sum()); tot += len(keep)
    out = pd.Series(final(subf.reset_index(drop=True), color, pc), index=subf.index)
    assert (out.reindex(keep).to_numpy() == full[keep]).all()
print(f'(2) CNN 포함: 자른 입력 8개, {tot:,}행 중 CNN 확률이 한 비트라도 달라진 행 {bad}개, 최종 예측 차이 0행', flush=True)
# (3) 학습 실험
tr = pd.concat([pd.read_csv(D / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'}) for s in ('train', 'validation')], ignore_index=True)
key = tr.groupby('experiment_id')[['bat_class', 'bat_cc', 'tr_method']].first()
pick = key.groupby(['bat_class', 'bat_cc', 'tr_method']).head(3).index
trs = tr[tr.experiment_id.isin(pick)].reset_index(drop=True)
C = Path.home() / 'jaehyeok' / 'cache'; col2 = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
sweep(trs, col2, np.full((len(trs), 6), 1 / 6), 15, f'(3) 학습 실험 {len(pick)}개 (구성 9종)')
