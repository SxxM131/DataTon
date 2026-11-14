"""17팀 최종 추론 스크립트 (학습·fit 없음, weights/ 로드만).

예) python test.py --input <battery_data_light>/test/test.csv \
                   --image_root <battery_data_light>/test \
                   --output submission.csv
    (선택) --sample_submission <battery_data_light>/sample_submission.csv  → 제출 양식의 행 순서로 저장
           --device auto|cuda|cpu (기본 auto: GPU가 있으면 GPU)

처리 순서
  1. 입력 CSV의 행만 읽는다. 이미지 경로 연결 → 이미지 색 통계(이미지마다 독립).
  2. 테이블 피처 266개 + 규칙 피처 30개: 실험별 relative_time_s 정렬 후 과거·현재 관측만 사용 (초 기준 윈도우).
  3. LightGBM 기본 모델(weights/tab_lgbm.joblib) → P_tab
     2·3단계 재판정 모델(같은 파일): 2의 피처 + 압력 상승 피처(과거 관측 + 학습 때 저장한 구성별 압력 범위 중앙값) → P_tab23
  4. pairdiff CNN(weights/cnn_pairdiff.pt): 현재 + 같은 실험의 직전 관측 프레임 + 차이 → P_cnn (1장씩, 결정적 설정)
  5. P ∝ P_tab^0.8 · P_cnn^0.2 → label = argmax + 1 (1~6 정수)
  6. label이 2 또는 3인 행만 P_tab23 의 p2·p3 중 큰 쪽으로 다시 정한다 (다른 단계는 그대로, 이미지 확률은 섞지 않음).
  7. label이 6인데 온도 합(t1+t2)의 현재 값이 지금까지의 최고점이면 (아직 내려오기 시작하지 않았으면) 5로 둔다.
  8. label이 5 이상인데 그 구성의 5단계 이벤트가 아직 관측되지 않았으면 4로 되돌린다
     (각형 28Ah 가열·각형 280Ah 과충전: 전압 급락 이후 온도 급상승 / 원통형: 두 번째 전압 강하).
  9. 원통형에서는 4단계에 들어선 뒤 30초가 지나기 전의 5단계 이상 판정을 4로 둔다.
 10. 같은 실험의 앞선 시점에 4단계(5단계) 이상으로 판정한 적이 있으면 그 아래로 내려가지 않게 한다 (4단계는 전압 급락을 관측한 뒤부터).
 11. 각형에서 label이 5인데 온도 급상승 뒤 온도 합(t1+t2)이 현재의 최고점보다 30℃ 넘게 내려온 적이 있으면 6으로 올린다.
     입력 CSV 행 순서로 저장 (index=False)
입력 행만으로 계산하므로 행 순서를 섞거나, 실험 일부만 넣거나, 실험별 앞부분만 넣어도 같은 행에는 같은 예측이 나온다.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '6')

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import model as M  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--weights_dir', type=Path, default=HERE / 'weights')
    parser.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--save_proba', type=Path, default=None, help='(선택) 융합 전후 확률 저장 경로 (.pkl)')
    args = parser.parse_args()
    t0 = time.time()

    import torch
    torch.set_num_threads(4)
    M.set_torch_deterministic()
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique, 'sample_id 중복'
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])

    # --- 테이블 모델 ---
    bundle = joblib.load(args.weights_dir / 'tab_lgbm.joblib')
    color = M.extract_image_features(frame, paths)
    X = M.build_features(frame, color)[bundle['columns']]
    p_tab = bundle['model'].predict_proba(X)
    X23 = pd.concat([X, M.pressure_features(frame, bundle['range_table'])], axis=1)[bundle['columns23']]
    p_tab23 = bundle['model23'].predict_proba(X23)

    # --- 이미지 CNN ---
    cnn = M.load_cnn(args.weights_dir / 'cnn_pairdiff.pt', dev)
    images = M.decode_images(paths)
    p_cnn = M.cnn_probs(frame, images, cnn, dev)

    # --- 후기 융합 ---
    p = M.fuse(p_tab, p_cnn, bundle['fuse_w'])
    label = M.apply_rules(M.to_label(p), p_tab23, frame)    # 2·3 재판정 → 6·5단계 근거 확인 → 원통형 4단계 최소 길이 → 되돌아가지 않기 → 6단계 진입(각형)
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
    print(f'저장: {args.output} ({len(submission)}행, device={dev}, {time.time() - t0:.1f}s) '
          f'분포 {submission.label.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
