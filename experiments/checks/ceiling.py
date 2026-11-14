"""v9 상한 분석 (분석 전용): 전환별 남은 손실, '한 관측 늦게 확인하는 이상적 탐지기'의 점수, 구성별 오류 위치와 방향.

사용: python experiments/checks/ceiling.py <레포 폴더>
마지막 부분은 규정을 어기고 미래 관측을 쓴 분석 전용 모델(experiments/stage3/noncausal_doc_oof.py의 OOF, 레포 밖)과 비교한다 — 제출과 무관.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
REPO = Path(sys.argv[1]).resolve(); sys.path[:0] = [str(REPO), str(REPO / 'experiments' / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402
D = Path.home() / 'data' / 'battery_data_light'
pool = cv.load_pool(D); evalp = cv.to_test_like(pool).copy(); y = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy()
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
a = [A[f'pred_r{r}'].to_numpy(int) for r in range(3)]; pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
P = [M.apply_rules(a[r], pb[r], evalp) for r in range(3)]
CLS = {1: '원통형', 2: '파우치', 3: '각형'}; TM = {1: '가열', 2: '과충전'}
cfg = (evalp.bat_class.map(CLS) + ' ' + evalp.bat_cc.map(lambda v: f'{v:g}Ah') + ' ' + evalp.tr_method.map(TM)).to_numpy()
te = pd.read_csv(D / 'test' / 'test.csv'); tcfg = (te.bat_class.map(CLS) + ' ' + te.bat_cc.map(lambda v: f'{v:g}Ah') + ' ' + te.tr_method.map(TM))
test_cfgs = te.assign(c=tcfg).groupby('c').experiment_id.nunique().to_dict(); in_test = np.isin(cfg, list(test_cfgs))
# 실험별 시간순 위치
order = np.lexsort((evalp.relative_time_s.to_numpy(float), e)); eo = e[order]; first = np.r_[True, eo[1:] != eo[:-1]]


def late(lab, k=1, only=None, keep12=False):
    """모든(또는 only) 전환을 k행 늦춘 라벨. keep12: 1→2 전환은 그대로."""
    out = lab.copy(); lo = lab[order]
    for _ in range(k):
        prev = np.where(first, lo, np.r_[lo[:1], lo[:-1]]); new = lo.copy()
        for b in range(1, 6):
            if only is not None and b != only:
                continue
            if keep12 and b == 1:
                continue
            m = (lo > b) & (prev <= b); new[m] = np.minimum(new[m], np.maximum(prev[m], b))
        lo = new
    out[order] = lo
    return out


def fixb(p, bs):
    q = p.copy()
    for b in bs:
        q = np.where((y <= b) & (q > b), b, q); q = np.where((y > b) & (q <= b), b + 1, q)
    return q


def mf(preds, m):
    return float(np.mean([f1_score(y[m], p[m], labels=range(1, 7), average='macro') for p in preds]))


def pc(preds, m):
    return np.mean([f1_score(y[m], p[m], labels=range(1, 7), average=None) for p in preds], axis=0)


ALL = np.ones(len(y), bool)
print(f'평가 행 {len(y)} (실험 {len(np.unique(e))}개), Test에 있는 구성의 실험 {len(np.unique(e[in_test]))}개 / 행 {int(in_test.sum())}')
print('Test 구성(실험 수):', test_cfgs)
for name, m in (('전체 97개 실험', ALL), ('Test에 있는 구성만', in_test)):
    print(f'\n=== {name} ===')
    print(f'v9 현재            {mf(P, m):.4f}  단계별 {np.round(pc(P, m), 3)}')
    for b in range(1, 6):
        nerr = np.mean([(((p <= b) != (y <= b)) & m).sum() for p in P])
        print(f'  {b}→{b + 1} 전환만 정답이면   {mf([fixb(p, [b]) for p in P], m):.4f} ({mf([fixb(p, [b]) for p in P], m) - mf(P, m):+.4f}) | 이 전환에서 틀린 행 {nerr:.0f}')
    print(f'  2→3만 빼고 전부 정답  {mf([fixb(p, [1, 3, 4, 5]) for p in P], m):.4f}')
    print(f'  3→4·4→5·5→6 그대로, 2→3만 정답  {mf([fixb(p, [2]) for p in P], m):.4f}')
    for k in (1, 2):
        print(f'이상적 탐지기: 2단계는 즉시, 나머지 전환은 {k}행 늦게  {mf([late(y, k, keep12=True)], m):.4f}  단계별 {np.round(pc([late(y, k, keep12=True)], m), 3)}')
    for b in range(2, 6):
        print(f'  {b}→{b + 1}만 1행 늦게(나머지 정답) {mf([late(y, 1, only=b)], m):.4f}')
    # 실험당 전환별 틀린 행 수 (v9 vs 1행 지연)
    rows = []
    for b in range(1, 6):
        ex_has = np.unique(e[m & (y > b)]); per = []
        for p in P:
            w = pd.Series((((p <= b) != (y <= b)) & m).astype(int)).groupby(e).sum(); per.append(w.reindex(ex_has).to_numpy())
        per = np.mean(per, axis=0)
        rows.append((f'{b}→{b + 1}', len(ex_has), per.mean(), np.median(per), (per <= 1).mean(), (per == 0).mean()))
    print(pd.DataFrame(rows, columns=['전환', '실험', '실험당 틀린 행(평균)', '중앙값', '1행 이하 비율', '0행 비율']).round(2).to_string(index=False))
# 구성 × 전환 오류 행
print('\n=== 구성별 전환 오류 행 (3회 평균) ===')
rows = []
for c in sorted(set(cfg)):
    m = cfg == c; r = {'구성': c, 'Test': test_cfgs.get(c, 0), '실험': len(np.unique(e[m])), '행': int(m.sum())}
    for b in range(1, 6):
        r[f'{b}→{b + 1}'] = round(float(np.mean([(((p <= b) != (y <= b)) & m).sum() for p in P])), 1)
    r['F1'] = round(mf(P, m), 3); rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
# 단계별 실험당 행 수 (Test식 행)
cnt = pd.crosstab(e[in_test], y[in_test]); print('\nTest 구성 실험의 단계별 행 수 중앙값:', cnt.replace(0, np.nan).median().round(1).to_dict())
tp = pd.read_csv(Path.home() / 'jaehyeok/DataTon/submissions/sub18_final_v9.csv').merge(te[['sample_id', 'experiment_id']]); ct = pd.crosstab(tp.experiment_id, tp.label)
print('Test 예측(sub18)의 실험당 단계별 행 수 중앙값:', ct.replace(0, np.nan).median().round(1).to_dict(), '| 실험당 행', int(te.groupby('experiment_id').size().median()))

# ---- (추가) 규정을 어기고 미래·실험 전체 정규화를 쓴 모델(분석 전용 OOF, 반복 0)로 2·3 판정을 바꾸면 ----
H = pd.read_pickle(Path.home() / 'jaehyeok' / 'cache' / 'noncausal_doc_oof_r0.pkl').reindex(evalp.index); h = H['pred_r0'].to_numpy(int)
p0 = P[0]; q = np.where(np.isin(p0, (2, 3)) & np.isin(h, (2, 3)), h, p0)


def late345(p, k=1):
    yl = late(y, k, keep12=True); r = p.copy()
    for b in (3, 4, 5):
        r = np.where((yl <= b) & (r > b), b, r); r = np.where((yl > b) & (r <= b), b + 1, r)
    return r


print('\n=== 미래 정보를 쓴 모델(규정 위반, 분석 전용)과 비교 — 반복 0 ===')
for name, m in (('전체', ALL), ('Test 구성', in_test)):
    e23 = lambda p: int((((p <= 2) != (y <= 2)) & m).sum())  # noqa: E731
    print(f'[{name}] v9 {mf([p0], m):.4f} (2→3 틀린 행 {e23(p0)}) | 미래 정보 모델 단독 {mf([h], m):.4f} (2→3 틀린 행 {e23(h)}) 단계별 {np.round(pc([h], m), 3)}')
    print(f'    v9에서 2·3 판정만 미래 정보 모델로 {mf([q], m):.4f} ({mf([q], m) - mf([p0], m):+.4f}, 2→3 틀린 행 {e23(q)})')
    print(f'    v9의 2→3은 그대로, 3→4·4→5·5→6을 "1행 늦은 정답"으로 {mf([late345(p0)], m):.4f} ({mf([late345(p0)], m) - mf([p0], m):+.4f}) | 2행 늦은 정답으로 {mf([late345(p0, 2)], m):.4f}')
    print(f'    둘 다 (미래 정보 2·3 + 1행 늦은 4~6) {mf([late345(q)], m):.4f}')
    print(f'    (3회 평균) v9 {mf(P, m):.4f} → 3→4·4→5·5→6을 1행 늦은 정답으로 {mf([late345(x) for x in P], m):.4f} / 2행 늦은 정답으로 {mf([late345(x, 2) for x in P], m):.4f}')
# 2→3: 실험별 틀린 행 수 분포와 구성
per = pd.Series(((p0 <= 2) != (y <= 2)).astype(int)).groupby(e).sum(); perh = pd.Series(((h <= 2) != (y <= 2)).astype(int)).groupby(e).sum()
print('2→3 실험당 틀린 행: v9 중앙값 %.0f 평균 %.1f, 상위 10개 실험이 전체의 %.0f%% | 미래 정보 모델 중앙값 %.0f 평균 %.1f' % (per.median(), per.mean(), 100 * per.sort_values().tail(10).sum() / per.sum(), perh.median(), perh.mean()))

# ---- (추가) Test 구성: 전환별 빠름/늦음 방향과 쏠림 (3회 평균) ----
print('\n=== Test 구성: 전환별 오류의 방향 (빠름 = 실제보다 먼저 넘김) ===')
rows = []
for c in sorted(test_cfgs):
    m = cfg == c; nex = len(np.unique(e[m]))
    for b in (2, 3, 4, 5):
        early = np.mean([((p > b) & (y <= b) & m).sum() for p in P]); lt = np.mean([((p <= b) & (y > b) & m).sum() for p in P])
        per = np.mean([pd.Series((((p <= b) != (y <= b)) & m).astype(int)).groupby(e).sum().reindex(np.unique(e[m])).to_numpy() for p in P], axis=0)
        hh = pd.Series((((h <= b) != (y <= b)) & m).astype(int)).groupby(e).sum().reindex(np.unique(e[m])).to_numpy()
        rows.append({'구성': c, '전환': f'{b}→{b + 1}', '실험': nex, '빠름': round(early, 1), '늦음': round(lt, 1), '실험당': round(per.mean(), 1), '중앙값': round(float(np.median(per)), 1),
                     '가장 큰 실험 1개 비중': f'{100 * per.max() / max(per.sum(), 1e-9):.0f}%', '1행 이하인 실험': f'{int((per <= 1).sum())}/{nex}', '미래 정보 모델 실험당': round(hh.mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
