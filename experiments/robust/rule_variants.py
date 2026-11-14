"""규칙 변형의 오프라인 평가 도구: 6단계 확인(최고점 판정) 변형, '되돌아가지 않기'의 4단계에 전압 급락 근거를 요구하는 변형."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import model as M  # noqa: E402


def _groups(frame):
    t = frame.relative_time_s.to_numpy(float); sid = frame.sample_id.astype(str).to_numpy()
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.astype(str).to_numpy(), sort=False).indices.items():
        yield idx[np.lexsort((sid[idx], t[idx]))]


def gate6_v(p, frame, delta=0.0, wait=0.0):
    """6 예측은 '최고점을 지났다'는 근거가 있을 때만: 누적 최고점보다 delta 넘게 내려왔거나, 최고점 갱신이 wait초 넘게 멈춤. 아니면 5."""
    out = np.asarray(p).copy(); t = frame.relative_time_s.to_numpy(float); s_all = (frame.meas_t1 + frame.meas_t2).to_numpy(float)
    for idx in _groups(frame):
        sm = s_all[idx]; cm = np.maximum.accumulate(sm)
        last = np.maximum.accumulate(np.where(sm >= cm - 1e-9, t[idx], -np.inf)); since = t[idx] - last
        passed = (cm - sm > delta) | (since > wait)
        if delta == 0.0 and wait == 0.0:
            passed = sm < cm - 1e-9                       # 새 최고점이 아닌 행만 6 허용
        out[idx] = np.where((out[idx] == 6) & ~passed, 5, out[idx])
    return out


def vdrop_seen(frame):
    out = np.zeros(len(frame), bool); v_all = frame.meas_v.to_numpy(float)
    for idx in _groups(frame):
        v = v_all[idx]; dv = np.r_[0.0, np.diff(v)]
        out[idx] = np.maximum.accumulate((dv <= M.V_DROP) | (v < M.V_LOW))
    return out


def hold_v(p, frame, need_vdrop=False):
    """4·5단계 되돌아가지 않기. need_vdrop=True면 4단계 유지에는 전압 급락 이벤트 관측을 요구한다 (근거 없는 4는 굳히지 않음)."""
    out = np.asarray(p).copy(); vd = vdrop_seen(frame) if need_vdrop else None
    for idx in _groups(frame):
        x = out[idx]
        seen4 = np.maximum.accumulate(x >= 4)
        if need_vdrop:
            seen4 = seen4 & vd[idx]
        x = np.where(seen4 & (x < 4), 4, x)
        x = np.where(np.maximum.accumulate(x >= 5) & (x < 5), 5, x)
        out[idx] = x
    return out


def chain(a, pb, frame, g6=(0.0, 0.0), g6_rate=None, need_vdrop=False):
    x = M.redecide_23(a, pb)
    x = M.gate_stage6(x, frame) if g6_rate is not None else gate6_v(x, frame, *g6)
    x = M.floor_stage4(M.gate_stage5(x, frame), frame)
    return hold_v(x, frame, need_vdrop)


if __name__ == '__main__':
    sys.path.insert(0, str(HERE.parent / 'common')); import cv
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    R = HERE.parent / 'results' / 'oof'
    A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
    y = evalp.label.to_numpy(int)
    a = [A[f'pred_r{r}'].to_numpy(int) for r in range(3)]; pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
    base = pd.DataFrame({f'pred_r{r}': M.apply_rules(a[r], pb[r], evalp) for r in range(3)}, index=evalp.index)
    r0 = cv.evaluate(evalp, base, n_boot=50); print(f"현재 체인 CV {r0['cv']:.4f} {np.round(r0['per_class'], 3).tolist()}")
    def show(name, **kw):
        g = pd.DataFrame({f'pred_r{r}': chain(a[r], pb[r], evalp, **kw) for r in range(3)}, index=evalp.index)
        c = cv.compare(evalp, g, base, n_boot=2000); rr = cv.evaluate(evalp, g, n_boot=50)
        print(f"{name:52s} CV {rr['cv']:.4f} ({c['diff']:+.4f}, p {c['p_a_better']:.2f}) 4·5·6 F1 {np.round(rr['per_class'][3:], 3).tolist()}", flush=True)
    show('6단계: 새 최고점이면 5 (속도 조건 없음)', g6=(0.0, 0.0))
    for dl, wt in ((0.5, 5), (1, 5), (1, 10), (2, 10), (2, 15), (3, 15), (3, 20), (5, 20), (5, 30)):
        show(f'6단계: 최고점보다 {dl}℃ 넘게 하강 또는 {wt}초 넘게 정체', g6=(float(dl), float(wt)))
    show('되돌아가지 않기: 4단계는 전압 급락 근거 필요', g6_rate=0.2, need_vdrop=True)
    show('위 둘: 새 최고점이면 5 + 4단계 근거 필요', g6=(0.0, 0.0), need_vdrop=True)
