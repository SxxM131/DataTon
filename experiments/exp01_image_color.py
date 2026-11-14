"""실험 01: 그레이스케일 이미지 통계 vs 팔레트 역변환(의사 온도) 이미지 통계.

열화상 이미지는 rainbow 계열 의사색상(마젠타=저온 → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색=고온).
그레이스케일 휘도는 이 팔레트에서 온도와 단조 관계가 아니므로, 색조(hue)를 의사 온도로 되돌린다.

모델은 베이스라인과 동일한 LogisticRegression 파이프라인으로 고정해 이미지 피처 효과만 비교한다.

예) python experiments/exp01_image_color.py --data_dir ~/data/battery_data_light --cache_dir <캐시 폴더>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as M  # noqa: E402

COLOR_STATS = ['pt_mean', 'pt_std', 'pt_p50', 'pt_p90', 'pt_p99', 'pt_max',
               'pt_frac20', 'pt_frac50', 'pt_frac70', 'pt_frac90', 'white_frac']


def color_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        hsv = np.asarray(rgb.convert('HSV'), dtype=np.float32)
    h = hsv[..., 0] * (360.0 / 256.0)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    white = (s < 0.16) & (v > 0.78)
    valid = (v > 0.15) | white                      # 너무 어두운 픽셀(테두리 등)은 제외
    pt = np.clip((300.0 - h) / 300.0, 0.0, 1.0)     # 마젠타(300°)=0 ... 빨강(0°)=1
    pt = np.where(h > 330.0, 1.0, pt)               # 330~360°는 빨강 쪽으로 간주
    pt = np.where(white, 1.0, pt)
    x = pt[valid] if valid.any() else pt.ravel()
    return [float(x.mean()), float(x.std()), *map(float, np.percentile(x, [50, 90, 99])), float(x.max()),
            float((x >= .2).mean()), float((x >= .5).mean()), float((x >= .7).mean()), float((x >= .9).mean()),
            float(white.mean())]


def load(data_dir: Path, split: str, cache_dir: Path):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cg, cc = cache_dir / f'{split}_gray.pkl', cache_dir / f'{split}_color.pkl'
    if not (cg.exists() and cc.exists()):
        paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
        M.extract_image_features(frame, paths).to_pickle(cg)
        rows = Parallel(n_jobs=-1, batch_size=64)(delayed(color_stats)(p) for p in paths)
        pd.DataFrame(rows, columns=COLOR_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id')).to_pickle(cc)
    return frame, pd.read_pickle(cg), pd.read_pickle(cc)


def first_frame_delta(frame: pd.DataFrame, feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """같은 실험의 가장 이른 관측(현재 시점 이전) 대비 변화량. 과거 값만 사용."""
    t = pd.to_numeric(frame['relative_time_s']).to_numpy()
    out = pd.DataFrame(index=frame.index, columns=[f'{c}_d0' for c in cols], dtype=float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        first = idx[np.argmin(t[idx])]
        out.iloc[idx] = feats.iloc[idx][cols].to_numpy() - feats.iloc[[first]][cols].to_numpy()
    return out


def make_X(frame, gray, color, mode):
    base = pd.DataFrame({'sample_id': frame.sample_id, 'experiment_id': frame.experiment_id,
                         'relative_time_s': pd.to_numeric(frame.relative_time_s)})
    base[M.SENSORS] = M.numeric(frame, M.SENSORS)
    parts = [base[['relative_time_s'] + M.SENSORS], M.add_causal_features(base, M.SENSORS)]
    if mode in ('gray', 'gray+color', 'gray+color+temporal'):
        g = gray.reindex(frame.sample_id.values).set_index(frame.index)
        base[list(g.columns)] = g
        parts += [g, M.add_causal_features(base, ['img_max', 'img_p99'])]
    if mode in ('color', 'gray+color', 'color+temporal', 'gray+color+temporal'):
        c = color.reindex(frame.sample_id.values).set_index(frame.index)
        parts.append(c)
        if mode.endswith('temporal'):
            key = ['pt_mean', 'pt_p90', 'pt_p99', 'pt_frac50', 'pt_frac70', 'white_frac']
            base[key] = c[key]
            parts += [M.add_causal_features(base, key), first_frame_delta(frame, c, key)]
    return pd.concat(parts, axis=1)


def mf1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--cache_dir', type=Path, required=True)
    args = ap.parse_args()
    tr, gtr, ctr = load(args.data_dir, 'train', args.cache_dir)
    va, gva, cva = load(args.data_dir, 'validation', args.cache_dir)
    ytr, yva = tr.label.astype(int), va.label.astype(int)

    for mode in ['sensor_only', 'gray', 'color', 'gray+color', 'color+temporal', 'gray+color+temporal']:
        Xtr, Xva = make_X(tr, gtr, ctr, mode), make_X(va, gva, cva, mode)
        cv = []
        for a, b in GroupKFold(5).split(Xtr, ytr, tr.experiment_id):
            cv.append(mf1(ytr.iloc[b], M.build_model().fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b])))
        m = M.build_model().fit(Xtr, ytr)
        p = m.predict(Xva)
        f1s = f1_score(yva, p, labels=M.LABELS, average=None)
        print(f'{mode:22s} n_feat={Xtr.shape[1]:3d} | CV {np.mean(cv):.4f}±{np.std(cv):.4f} | Val {mf1(yva, p):.4f} | '
              f'per-class {np.round(f1s, 3).tolist()}', flush=True)


if __name__ == '__main__':
    main()
