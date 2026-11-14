"""센서·이미지 결합 방식 4가지 비교 (팀원 제안: 센서 단독 / 이미지 단독 / 후기 융합 / 특징 수준 융합).

같은 fold(공통 검증 rep 0 — pairdiff CNN의 OOF 확률이 있는 회차)에서 기존 OOF 확률만으로 계산한다 (재학습 없음).
  - 센서(테이블) 모델: rules_tab_nocum_r3, 2·3단계 재판정 모델: st3_presrise_mask_r3
  - 이미지 모델: pairdiff CNN (~/jaehyeok/cache/img/cnn_pairdiff.npz)
  - 후기 융합: 산술 평균 α·P_sensor + (1−α)·P_image (팀원 제안식), 기하 평균 P_sensor^(1−w)·P_image^w (최종 모델 방식)
  - α를 평가 fold 밖(나머지 4개 fold)에서 고르는 중첩 선택 포함
특징 수준 융합(CNN 확률을 테이블 모델 입력에 추가)은 stack.py 결과(img_stack_tab_single_probs 0.8939, 같은 피처의 센서 단독 0.8958)를 참고.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); y = evalp.label.to_numpy(int)
pos = pool.index.get_indexer(evalp.index); fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
fo = fold_of[pos]
R = HERE.parent / 'results' / 'oof'; cols = [f'p{k}_r0' for k in range(1, 7)]
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
with np.load(Path.home() / 'jaehyeok' / 'cache' / 'img' / 'cnn_pairdiff.npz') as z:
    Cn = np.zeros((len(evalp), 6))
    for k in range(5):
        m = fo == k; Cn[m] = z[f'out{k}_eval'][pos[m]]
O = lambda p: pd.DataFrame({'pred_r0': np.asarray(p).astype(int)}, index=evalp.index)  # noqa: E731


def line(name, p, ref=None):
    r = cv.evaluate(evalp, O(p), n_boot=100)
    s = f"{name:44s} CV {r['cv']:.4f} {np.round(r['per_class'], 3).tolist()} | 3을 2로 {int(((y == 3) & (p == 2)).sum()):4d}행, 2를 3으로 {int(((y == 2) & (p == 3)).sum()):4d}행"
    if ref is not None:
        c = cv.compare(evalp, O(p), O(ref), n_boot=2000); s += f" | 센서 모델 대비 {c['diff']:+.4f} (p {c['p_a_better']:.2f})"
    print(s, flush=True)


ta = M.to_label(A)
line('1. 센서(테이블) 모델 단독', ta)
line('2. 이미지(CNN) 모델 단독', M.to_label(Cn), ta)
for al in (0.9, 0.8, 0.7, 0.5):
    line(f'3. 후기 융합, 산술 평균 α={al} (센서 비중)', M.to_label(al * A + (1 - al) * Cn), ta)
line('3. 후기 융합, 기하 평균 w=0.2 (최종 모델 방식)', M.to_label(M.fuse(A, Cn, 0.2)), ta)
AL = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5); pred = np.zeros(len(y), int); chosen = []
for k in range(5):
    tr, va = fo != k, fo == k
    best = max(AL, key=lambda a: cv.macro_f1(y[tr], M.to_label(a * A[tr] + (1 - a) * Cn[tr]))); chosen.append(best)
    pred[va] = M.to_label(best * A[va] + (1 - best) * Cn[va])
line(f'3. 산술 평균, α를 fold 밖에서 선택 {chosen}', pred, ta)
print('--- 최종 규칙 보정(2·3 재판정, 5·6단계 근거 확인, 되돌아가지 않기)까지 적용한 뒤 ---')
fa = M.apply_rules(ta, B, evalp)
line('센서 모델 + 규칙 보정', fa)
line('후기 융합(w=0.2) + 규칙 보정 (최종 구성)', M.apply_rules(M.to_label(M.fuse(A, Cn, 0.2)), B, evalp), fa)
