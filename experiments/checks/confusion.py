"""최종 구성(테이블 모델 + 2·3단계 재판정 + 규칙 보정)의 교차검증 혼동행렬 (분석 전용).

사용: python experiments/checks/confusion.py <레포 폴더>
5-fold × 3회 교차검증의 OOF 예측(레포에 올리지 않는 experiments/results/oof/)에 model.apply_rules를 적용하고,
Test식 다운샘플 행에서 센다. 값은 3회 반복의 평균이다. CNN 융합은 포함하지 않는다 (README의 0.9229와 같은 기준).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
REPO = Path(sys.argv[1]).resolve(); sys.path[:0] = [str(REPO), str(REPO / 'experiments' / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

D = Path.home() / 'data' / 'battery_data_light'
pool = cv.load_pool(D); evalp = cv.to_test_like(pool).copy(); y = evalp.label.to_numpy(int)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
P = [M.apply_rules(A[f'pred_r{r}'].to_numpy(int), B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(), evalp) for r in range(3)]

L = list(range(1, 7))
cm = np.mean([confusion_matrix(y, p, labels=L) for p in P], axis=0)
f1 = np.mean([f1_score(y, p, labels=L, average=None) for p in P], axis=0)
macro = float(np.mean([f1_score(y, p, labels=L, average='macro') for p in P]))
rec = np.diag(cm) / cm.sum(1); prec = np.diag(cm) / cm.sum(0)
print(f'평가 행 {len(y)} (실험 {evalp.experiment_id.nunique()}개), Macro F1 {macro:.4f}')
print('\n혼동행렬 (행 = 실제 단계, 열 = 예측 단계, 3회 평균 행 수)')
print(pd.DataFrame(cm, index=[f'실제 {k}' for k in L], columns=[f'예측 {k}' for k in L]).round(1).to_string())
print('\n행 기준 비율 (%)')
print(pd.DataFrame(100 * cm / cm.sum(1, keepdims=True), index=[f'실제 {k}' for k in L], columns=[f'예측 {k}' for k in L]).round(1).to_string())
print('\n단계별 정밀도 / 재현율 / F1')
print(pd.DataFrame({'행 수': cm.sum(1).round(0).astype(int), '정밀도': prec.round(3), '재현율': rec.round(3), 'F1': f1.round(3)}, index=[f'{k}단계' for k in L]).to_string())
err = cm.sum() - np.trace(cm)
adj = sum(cm[i, j] for i in range(6) for j in range(6) if abs(i - j) == 1)
early = sum(cm[i, j] for i in range(6) for j in range(6) if j == i + 1); late = sum(cm[i, j] for i in range(6) for j in range(6) if j == i - 1)
print(f'\n틀린 행 {err:.1f} ({100 * err / cm.sum():.2f}%). 그중 바로 옆 단계로 틀린 행 {adj:.1f} ({100 * adj / err:.1f}%): '
      f'다음 단계로 일찍 넘긴 행 {early:.1f}, 이전 단계에 늦게 남은 행 {late:.1f}')
pair = {f'{i + 1}↔{i + 2}': cm[i, i + 1] + cm[i + 1, i] for i in range(5)}
print('경계별 틀린 행: ' + ', '.join(f'{k} {v:.1f} ({100 * v / err:.1f}%)' for k, v in pair.items()))
