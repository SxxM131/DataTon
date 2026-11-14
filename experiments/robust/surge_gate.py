"""'온도 급상승 근거' 게이트의 공통 검증.

가이드상 각형 28Ah 가열·각형 280Ah 과충전의 5단계 = 4단계(전압 급락) 이후 온도가 급상승하는 지점.
학습 실험 23개 전부에서 '전압 급락 이후 10초 온도 상승 ≥ 임계'는 5단계 시작 +0~+2행에 처음 참이 되고, 4단계 동안에는 최대 9.6℃였다.
게이트: 해당 구성에서 5단계로 예측했는데 급상승이 아직 관측되지 않았으면 4단계로 되돌린다 (과거 관측만 사용, 상수 임계).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402


def surge_seen(frame: pd.DataFrame, thr: float = 12.0, w: float = 10.0, rate_thr: float | None = 1.0) -> np.ndarray:
    """전압 급락 이벤트 이후 온도 급상승을 관측한 적이 있는가 (frame 행 순서, 과거 관측만).

    급상승 = (직전 관측 대비 초당 상승 ≥ rate_thr) 또는 (w초 이하 전 마지막 관측 대비 상승 ≥ thr). 비교 시점도 급락 이후여야 한다.
    """
    out = np.zeros(len(frame), bool)
    t_all = frame.relative_time_s.to_numpy(float); sid = frame.sample_id.astype(str).to_numpy()
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        t = t_all[idx]; t1 = frame.meas_t1.to_numpy(float)[idx]; v = frame.meas_v.to_numpy(float)[idx]
        dv = np.r_[0.0, np.diff(v)]
        ev = (dv <= M.V_DROP) | (v < M.V_LOW)
        if not ev.any():
            continue
        ivd = int(np.argmax(ev))
        j = np.searchsorted(t, t - w, side='right') - 1
        ok = j >= ivd
        r = np.zeros(len(t)); r[ok] = t1[ok] - t1[j[ok]]
        hit = r >= thr
        if rate_thr is not None:
            with np.errstate(invalid='ignore', divide='ignore'):
                rate = np.r_[0.0, np.diff(t1) / np.diff(t)]
            rate[:ivd + 1] = 0.0
            hit = hit | (np.nan_to_num(rate, nan=0.0) >= rate_thr)
        out[idx] = np.maximum.accumulate(hit)
    return out


if __name__ == '__main__':
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    R = HERE.parent / 'results' / 'oof'
    A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
    y = evalp.label.to_numpy(int)
    base = pd.DataFrame(index=evalp.index)
    for r in range(3):
        a = A[f'pred_r{r}'].to_numpy(int)
        base[f'pred_r{r}'] = M.redecide_23(a, B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy())
    cfgs = {'각형 28Ah 가열': (evalp.bat_class == 3) & (evalp.bat_cc == 28) & (evalp.tr_method == 1), '각형 280Ah 과충전': (evalp.bat_class == 3) & (evalp.bat_cc == 280) & (evalp.tr_method == 2)}
    target = (cfgs['각형 28Ah 가열'] | cfgs['각형 280Ah 과충전']).to_numpy()
    r0 = cv.evaluate(evalp, base, n_boot=200)
    print(f"현재 v2(테이블) CV {r0['cv']:.4f} per-class {np.round(r0['per_class'], 3).tolist()}")
    for thr, rt in ((12.0, None), (12.0, 0.84), (12.0, 1.0), (15.0, 1.0)):
        seen = surge_seen(evalp, thr, rate_thr=rt)
        for mode in ('5→4', '5→4 & 4→5'):
            g = base.copy(); ch = []
            for r in range(3):
                p = g[f'pred_r{r}'].to_numpy().copy()
                m = target & (p == 5) & ~seen; p[m] = 4
                if mode != '5→4':
                    m2 = target & (p == 4) & seen; p[m2] = 5; m = m | m2
                ch.append((int(m.sum()), int((m & (p == y)).sum()), int((m & (base[f'pred_r{r}'].to_numpy() == y)).sum())))
                g[f'pred_r{r}'] = p
            c = cv.compare(evalp, g, base, n_boot=3000); rr = cv.evaluate(evalp, g, n_boot=100)
            ch = np.mean(ch, 0)
            print(f"임계 {thr:4.0f}℃/10초, 초당 {rt} [{mode:10s}] CV {rr['cv']:.4f} ({c['diff']:+.4f}, p {c['p_a_better']:.2f}) | 4·5단계 F1 {rr['per_class'][3]:.3f}/{rr['per_class'][4]:.3f}"
                  f" | 바뀐 행 {ch[0]:.0f} (바꿔서 맞음 {ch[1]:.0f}, 원래 맞았음 {ch[2]:.0f})", flush=True)
