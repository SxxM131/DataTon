"""[라운드 2-1] 단계 4/5/6 시작 행에서 이미지가 (센서보다 먼저) 변하는가 — 학습 없는 직접 점검.

프레임 변화량 d[i] = mean|pt32[i] - pt32[직전 관측]| (의사온도 32×32) 및 RGB 버전.
인과적 정규화: r[i] = d[i] / (t1>=45 이후 ~ i-1 까지의 d 최댓값)  (2단계 시작 전의 노이즈 프레임 제외; 2단계 시작은 t1>=45로 인과적으로 알 수 있음)
전압 이벤트(세션 D 규칙): A|B = (5초 전압 차분 <= -0.56) | (v < 2.5) 의 첫 발화 행.
"""
import sys
import numpy as np
import pandas as pd
import imglib as L

pool, evalp = L.load_all()
PT = L.load_images(pool, 'pt', 128, mmap=False)
P = PT.reshape(len(PT), 32, 4, 32, 4).mean((2, 4), dtype=np.float32).reshape(len(PT), -1) / 255.0
RGB = L.load_images(pool, 'rgb', 128, mmap=False)
R = RGB.reshape(len(RGB), 32, 4, 32, 4, 3).mean((2, 4), dtype=np.float32).reshape(len(RGB), -1) / 255.0
del PT, RGB
et = L.cv.exp_table(pool)
rows = []
for e, idx in pool.groupby('experiment_id', sort=True).indices.items():
    idx = idx[np.argsort(pool.relative_time_s.values[idx], kind='mergesort')]
    lab = pool.label.values[idx]; v = pool.meas_v.values[idx]; t1 = pool.meas_t1.values[idx]
    d = np.r_[0, np.abs(P[idx[1:]] - P[idx[:-1]]).mean(1)]
    dr = np.r_[0, np.abs(R[idx[1:]] - R[idx[:-1]]).mean(1)]
    frac = np.r_[0, (np.abs(P[idx[1:]] - P[idx[:-1]]) > 0.15).mean(1)]          # 변화한 영역 비율
    s2 = int(np.argmax(t1 >= 45)) if (t1 >= 45).any() else 0
    def runmax(x):                                                               # i 이전(s2+1 이후) 최댓값
        y = x.copy(); y[:s2 + 2] = 0
        m = np.maximum.accumulate(y); return np.r_[0, m[:-1]]
    rm, rmr = runmax(d), runmax(dr)
    dv = np.r_[0, np.diff(v)]
    ev = (dv <= -0.56) | (v < 2.5)
    vfire = int(np.argmax(ev)) if ev.any() else -999
    for s in (4, 5, 6):
        if not (lab == s).any():
            continue
        o = int(np.argmax(lab == s))
        rec = dict(e=e, stage=s, o=o, n=len(idx), bat=et.bat_class[e], cc=et.bat_cc[e], tr=et.tr_method[e], prev_lab=lab[o - 1], v_off=vfire - o)
        for k in (-2, -1, 0, 1, 2):
            j = o + k
            if 0 <= j < len(idx):
                rec[f'r{k:+d}'] = d[j] / (rm[j] + 1e-4); rec[f'd{k:+d}'] = d[j]; rec[f'rr{k:+d}'] = dr[j] / (rmr[j] + 1e-4); rec[f'f{k:+d}'] = frac[j]
        # 인과적 검출기: s2 이후 r > TH 가 처음 발화하는 행 (onset 대비 오프셋). 3단계 이후로 제한한 버전도.
        for th in (1.5, 3.0):
            r = d / (rm + 1e-4); r[:s2 + 2] = 0
            fire = np.where(r > th)[0]
            near = fire[(fire >= o - 2) & (fire <= o + 2)]
            rec[f'fire{th}_before'] = int((fire < o - 2).sum()); rec[f'fire{th}_near'] = (int(near[0]) - o) if len(near) else np.nan
        rows.append(rec)
D = pd.DataFrame(rows)
D.to_pickle(L.ICACHE / 'onset_check.pkl')
pd.set_option('display.width', 250)
for s in (4, 5, 6):
    g = D[D.stage == s]
    print(f'\n===== 단계 {s} 시작 ({len(g)}개 실험) =====')
    print('정규화 변화량 r (중앙값) onset-2..+2:', [round(g[f"r{k:+d}"].median(), 2) for k in (-2, -1, 0, 1, 2)],
          '| r>1(과거 최대 갱신) 비율:', [round((g[f"r{k:+d}"] > 1).mean(), 2) for k in (-2, -1, 0, 1, 2)])
    print('원 변화량 d (중앙값):', [round(g[f"d{k:+d}"].median(), 4) for k in (-2, -1, 0, 1, 2)], '| 변화 영역 비율(중앙값):', [round(g[f"f{k:+d}"].median(), 3) for k in (-2, -1, 0, 1, 2)])
    if s == 4:
        for name, m in [('센서 규칙 정확(v_off=0)', g.v_off == 0), ('라벨이 1행 빠름(v_off=+1)', g.v_off == 1), ('그 외', ~g.v_off.isin([0, 1]))]:
            h = g[m]
            print(f'  [{name}] {len(h)}개: r 중앙값 {[round(h[f"r{k:+d}"].median(), 2) for k in (-1, 0, 1)]} | onset 행 r>1 {int((h["r+0"] > 1).sum())}/{len(h)}, r>1.5 {int((h["r+0"] > 1.5).sum())}, r>3 {int((h["r+0"] > 3).sum())}'
                  f' | onset-1 행 r>1.5 {int((h["r-1"] > 1.5).sum())} | onset+1 행 r>1.5 {int((h["r+1"] > 1.5).sum())}')
    for th in (1.5, 3.0):
        near = g[f'fire{th}_near']
        print(f'  검출기 r>{th}: onset±2 안 첫 발화 오프셋 분포 {near.value_counts().sort_index().to_dict()} (없음 {int(near.isna().sum())}) | onset-2 이전 오발화 횟수 중앙값 {g[f"fire{th}_before"].median():.0f}, 0회인 실험 {(g[f"fire{th}_before"] == 0).sum()}')
g = D[(D.stage == 4) & (D.v_off == 1)]
print('\n라벨이 1행 빠른 실험 목록:'); print(g[['e', 'bat', 'cc', 'tr', 'o', 'r-1', 'r+0', 'r+1', 'd-1', 'd+0', 'd+1', 'f+0', 'f+1', 'fire1.5_before']].round(3).to_string())
