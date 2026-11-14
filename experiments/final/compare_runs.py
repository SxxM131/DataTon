"""검증 D/E/CPU/검은 이미지 결과 비교 (run_bde.sh 출력물)."""
from pathlib import Path
import numpy as np
import pandas as pd
C = Path.home() / 'jaehyeok/cache/final'
R = Path(__file__).resolve().parents[2]
L = lambda n: pd.read_csv(C / f'{n}.csv', dtype={'sample_id': 'string'})
P = lambda n: pd.read_pickle(C / f'{n}_proba.pkl')
r1, r2 = L('run1'), L('run2')
print('D 두 번 실행 파일 동일:', (C / 'run1.csv').read_bytes() == (C / 'run2.csv').read_bytes())
a, b = P('run1'), P('run2')
print('  확률 비트 동일 (tab/cnn/fused):', [np.array_equal(a[k], b[k]) for k in ('p_tab', 'p_cnn', 'p')])
w2 = P('run_w2')
print('E train.py 2회 학습 → Test 테이블 확률 비트 동일:', np.array_equal(a['p_tab'], w2['p_tab']), ' 최대차', np.abs(a['p_tab'] - w2['p_tab']).max(),
      ' 최종 라벨 동일:', (L('run_w2').label == r1.label).all())
cpu = P('run_cpu')
print('CPU vs GPU: 라벨 다른 비율', (L('run_cpu').label != r1.label).mean(), ' cnn 확률 최대차', np.abs(cpu['p_cnn'] - a['p_cnn']).max(),
      ' 테이블 확률 동일', np.array_equal(cpu['p_tab'], a['p_tab']), ' 융합 최대차', np.abs(cpu['p'] - a['p']).max())
bl = L('run_black')
print('검은 이미지: 예측 바뀐 비율', (bl.label != r1.label).mean(), f'({(bl.label != r1.label).sum()}/{len(bl)})',
      ' 테이블 확률 최대차', np.abs(P('run_black')['p_tab'] - a['p_tab']).max())
tab_only = a['p_tab'].argmax(1) + 1
print('융합이 테이블 단독과 다른 비율(Test):', (tab_only != r1.label.to_numpy()).mean())
s8 = pd.read_csv(R / 'submissions/sub08_tab_all_nocum.csv', dtype={'sample_id': 'string'})
assert (s8.sample_id == r1.sample_id).all()
print('sub08 대비 다른 행 비율:', (s8.label != r1.label).mean(), f'({(s8.label != r1.label).sum()}행)', ' 테이블 단독 vs sub08:', (s8.label.to_numpy() != tab_only).mean())
ss = pd.read_csv(Path.home() / 'data/battery_data_light/sample_submission.csv', dtype={'sample_id': 'string'})
print('sample_submission 순서 = 입력 순서:', (ss.sample_id == r1.sample_id).all())
print('라벨 분포:', r1.label.value_counts().sort_index().to_dict())
