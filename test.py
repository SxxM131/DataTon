"""datathon_img 추론 (학습·fit 없음). 선택안 C: img10 표피처 + CNN 융합 w=0.1 + v7 게이트.

예) python test.py --input <battery_data_light>/test/test.csv \
                   --image_root <battery_data_light>/test \
                   --output submission.csv
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

import model as M  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--weights_dir', type=Path, default=HERE / 'weights')
    parser.add_argument('--sample_submission', type=Path, default=None)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--save_proba', type=Path, default=None)
    args = parser.parse_args()
    t0 = time.time()

    import torch
    torch.set_num_threads(4)
    M.set_torch_deterministic()
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique, 'sample_id 중복'
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])

    bundle = joblib.load(args.weights_dir / 'tab_lgbm.joblib')
    color = M.extract_image_features(frame, paths)
    mode = bundle.get('feature_mode', 'full')
    if mode == 'img10':
        X = M.build_features_img10(frame, color, bundle['non_img'])[bundle['columns']]
    else:
        X = M.build_features(frame, color)[bundle['columns']]
    p_tab = bundle['model'].predict_proba(X)
    X23 = pd.concat([X, M.pressure_features(frame, bundle['range_table'])], axis=1)[bundle['columns23']]
    p_tab23 = bundle['model23'].predict_proba(X23)

    cnn = M.load_cnn(args.weights_dir / 'cnn_pairdiff.pt', dev)
    images = M.decode_images(paths)
    p_cnn = M.cnn_probs(frame, images, cnn, dev)

    w = float(bundle.get('fuse_w', M.FUSE_W))
    p = M.fuse(p_tab, p_cnn, w)
    label = M.apply_rules(M.to_label(p), p_tab23, frame)
    pred = pd.Series(label, index=frame.sample_id.values)

    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order) == set(pred.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].values.astype(int)})
    assert submission.label.isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    if args.save_proba is not None:
        pd.to_pickle({'sample_id': frame.sample_id.tolist(), 'p_tab': p_tab, 'p_tab23': p_tab23, 'p_cnn': p_cnn, 'p': p}, args.save_proba)
    print(f'저장: {args.output} ({len(submission)}행, device={dev}, mode={mode}, fuse_w={w}, {time.time() - t0:.1f}s) '
          f'분포 {submission.label.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
