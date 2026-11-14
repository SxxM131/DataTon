"""17팀 최종 학습 스크립트.

예) python train.py --data_dir <battery_data_light 경로>
    → (1) 평가: Train 78개 실험으로 학습 → Validation 19개 실험 Macro F1 (테이블 단독 / 2·3단계 재판정 / 이미지 융합)
      (2) 최종 학습: Train + Validation 97개 실험 전체 → weights/tab_lgbm.joblib
          (기본 모델 + 2·3단계 재판정 모델 + 구성별 압력 범위 중앙값 표)
      (3) CNN 가중치: 기본은 제출에 쓴 학습 완료 가중치(--cnn_source)를 weights/cnn_pairdiff.pt 로 복사.
          --train_cnn 을 주면 PNG를 직접 읽어 97개 실험 전체로 다시 학습 (GPU 학습은 비트 재현이 안 되므로
          제출 결과와 정확히 같은 가중치는 레포에 들어 있는 weights/cnn_pairdiff.pt 이다).

옵션
  --skip_eval        평가 단계 생략
  --eval_cnn         평가 단계에서 Train만으로 CNN을 새로 학습해 융합 점수까지 계산 (GPU 필요, 약 1~2분)
  --train_cnn        최종 CNN을 새로 학습 (GPU 필요)

검증 행 구성: Test는 500행이 넘는 실험을 500행으로 균등 다운샘플해 두었으므로
  (a) Validation 원본 행 그대로, (b) Test와 같은 방식으로 다운샘플한 행 — 두 가지 점수를 모두 출력한다.
  어느 쪽이든 검증 행의 피처는 그 검증 행들만 보고 만든다 (Test에서 보이는 행과 같은 조건).
"""
import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '6')

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import model as M  # noqa: E402

HERE = Path(__file__).resolve().parent
CAP = 500                                      # Test의 실험당 최대 행 수
DEFAULT_CNN_SOURCE = Path.home() / 'jaehyeok/cache/img/cnn_pairdiff_full.pt'


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def load_pool(data_dir: Path) -> pd.DataFrame:
    """Train → Validation 순서로 이어 붙인 라벨 있는 전체 행 (원본 5초 간격). 이미지 실제 경로를 image_file 열로."""
    frames = []
    for split in ('train', 'validation'):
        f = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        f['origin'] = split
        f['image_file'] = M.resolve_image_paths(f, [data_dir / split, data_dir]).astype(str).values
        frames.append(f)
    pool = pd.concat(frames, ignore_index=True)
    assert pool.sample_id.is_unique
    assert set(pool.loc[pool.origin == 'train', 'experiment_id']).isdisjoint(pool.loc[pool.origin == 'validation', 'experiment_id'])
    return pool


def to_test_like(frame: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    """실험별 시간순 정렬 후 cap행을 넘으면 Test와 같은 규칙(floor(linspace(0, N-1, cap)))으로 다운샘플."""
    keep = np.zeros(len(frame), dtype=bool)
    t = frame['relative_time_s'].to_numpy()
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        if len(idx) > cap:
            idx = idx[np.floor(np.linspace(0, len(idx) - 1, cap)).astype(int)]
        keep[idx] = True
    return frame[keep]


def macro_f1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def per_class(y, p):
    return np.round(f1_score(y, p, labels=M.LABELS, average=None, zero_division=0), 3).tolist()


# ---------------------------------------------------------------------------
# CNN 학습 (experiments/img/finetune2.py --tag pairdiff 와 같은 설정, 이미지는 PNG에서 직접 디코딩)
# ---------------------------------------------------------------------------
def lag_context(pool: pd.DataFrame, lags=(5, 10, 15)) -> dict:
    """각 행의 W초 이전 프레임 위치: 같은 실험에서 t <= t_i - W 인 마지막 관측 (없으면 실험 첫 관측). 과거만."""
    n = len(pool)
    t = pool['relative_time_s'].to_numpy(dtype=float)
    out = {w: np.full(n, -1) for w in lags}
    for _, idx in pool.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        tt = t[idx]
        for w in lags:
            j = np.searchsorted(tt, tt - w, side='right') - 1
            out[w][idx] = idx[np.maximum(j, 0)]
    return out


def train_cnn(pool: pd.DataFrame, images: np.ndarray, train_mask: np.ndarray, seed: int,
              epochs=8, epoch_size=24000, bs=128, lr=5e-4):
    """현재 + 직전(5/10/15초 전 중 무작위) 프레임 + 차이 9채널 CNN. 클래스 가중 1/n, 긴 실험은 500행 상한 비중."""
    import torch
    import torch.nn.functional as F
    dev = 'cuda'
    torch.set_num_threads(4)
    N = len(pool)
    y = torch.tensor(pool.label.to_numpy(int) - 1, device=dev)
    T = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    ctx = lag_context(pool)
    P_prev = torch.stack([T(ctx[w]) for w in (5, 10, 15)])
    X = torch.from_numpy(images).to(dev)
    MEAN = torch.tensor(M.IMNET_MEAN * 2, device=dev).view(1, 6, 1, 1)
    STD = torch.tensor(M.IMNET_STD * 2, device=dev).view(1, 6, 1, 1)
    n_exp = pool.groupby('experiment_id')['label'].transform('size').to_numpy()
    base_w = np.minimum(1.0, CAP / n_exp)

    def batch(idx):
        prev = P_prev[torch.randint(0, 3, idx.shape, device=dev), idx]
        x = torch.cat([X[idx], X[prev]], -1).permute(0, 3, 1, 2).float().div_(255)
        B = len(idx)                                   # 증강: 크기·좌우 반전·이동 (현재·직전 프레임에 같은 변환)
        s = 1 + (torch.rand(B, device=dev) - .5) * .3
        flip = (torch.rand(B, device=dev) < .5).float() * 2 - 1
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = s * flip; th[:, 1, 1] = s * (1 + (torch.rand(B, device=dev) - .5) * .1)
        th[:, :, 2] = (torch.rand(B, 2, device=dev) - .5) * .2
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode='bilinear', padding_mode='border', align_corners=False)
        diff = (x[:, :3] - x[:, 3:]) * 2
        x = torch.cat([(x - MEAN) / STD, diff], 1)
        return x.contiguous(memory_format=torch.channels_last)

    torch.manual_seed(seed); np.random.seed(seed)
    m = M.make_cnn(pretrained=True).to(dev).to(memory_format=torch.channels_last)
    idx_all = np.where(train_mask)[0]
    yy = pool.label.to_numpy(int)[idx_all]
    cw = 1.0 / np.bincount(yy, minlength=7)[1:].clip(1)
    w = torch.tensor(base_w[idx_all] * cw[yy - 1], device=dev, dtype=torch.float)
    idx_all = T(idx_all)
    steps = epochs * (epoch_size // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    m.train()
    g = torch.Generator(device=dev); g.manual_seed(seed)
    for _ in range(steps):
        idx = idx_all[torch.multinomial(w, bs, replacement=True, generator=g)]
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(m(batch(idx)), y[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    del X
    return m.eval()


def save_cnn_fp16(m, path: Path):
    import torch
    torch.save({n: v.half() if v.is_floating_point() else v for n, v in m.state_dict().items()}, path)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    ap.add_argument('--weights_dir', type=Path, default=HERE / 'weights')
    ap.add_argument('--skip_eval', action='store_true')
    ap.add_argument('--eval_cnn', action='store_true', help='평가 단계에서 Train만으로 CNN 학습 → 융합 점수 (GPU)')
    ap.add_argument('--train_cnn', action='store_true', help='최종 CNN을 97개 실험 전체로 새로 학습 (GPU)')
    ap.add_argument('--cnn_source', type=Path, default=DEFAULT_CNN_SOURCE, help='복사할 학습 완료 CNN 가중치')
    args = ap.parse_args()
    t0 = time.time()
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.data_dir)
    color = M.extract_image_features(pool, pool['image_file'].tolist())     # 이미지마다 독립 → 행 순서 무관
    print(f'데이터 {len(pool)}행 / 실험 {pool.experiment_id.nunique()}개, 색 통계 {time.time() - t0:.0f}s', flush=True)
    y = pool.label.to_numpy(int)
    Xp = M.build_features(pool, color)
    columns = list(Xp.columns)
    columns23 = columns + M.Q_COLS                                          # 2·3단계 재판정 모델의 입력 열
    need_images = args.eval_cnn or args.train_cnn
    images = M.decode_images(pool['image_file'].tolist()) if need_images else None

    # ---------------- (1) 평가: Train → Validation ----------------
    scores = {}
    if not args.skip_eval:
        tr = (pool.origin == 'train').to_numpy()
        m = M.build_model().fit(Xp[tr], y[tr])
        er_tr = M.pressure_ranges(pool[tr])                                 # 압력 범위 통계도 Train 실험에서만 산출
        table_tr = M.range_table(er_tr)
        Q_tr = M.pressure_features(pool[tr], table_tr, own_range=M.loo_ranges(er_tr))
        m23 = M.build_model().fit(pd.concat([Xp[tr], Q_tr], axis=1)[columns23], y[tr])
        cnn = None
        if args.eval_cnn:
            import torch
            t1 = time.time()
            cnn = train_cnn(pool, images, tr, seed=100)
            M.set_torch_deterministic()
            print(f'  Train만으로 CNN 학습 {time.time() - t1:.0f}s', flush=True)
        val = pool[pool.origin == 'validation']
        for name, ev in [('raw', val), ('test_like', to_test_like(val))]:
            Xe = M.build_features(ev, color)[columns]              # 검증 행만 보고 피처 생성
            p_tab = m.predict_proba(Xe)
            p_tab23 = m23.predict_proba(pd.concat([Xe, M.pressure_features(ev, table_tr)], axis=1)[columns23])
            ye = ev.label.to_numpy(int)
            redo = M.gate_stage5(M.redecide_23(M.to_label(p_tab), p_tab23), ev)   # 2·3단계 재판정 + 5단계 근거 확인 (이미지 융합 없이)
            s = {'n_rows': len(ev), 'tab': macro_f1(ye, M.to_label(p_tab)), 'tab_per_class': per_class(ye, M.to_label(p_tab)),
                 'tab_redo23': macro_f1(ye, redo), 'tab_redo23_per_class': per_class(ye, redo)}
            if cnn is not None:
                pos = pool.index.get_indexer(ev.index)
                p_cnn = M.cnn_probs(ev, images[pos], cnn, 'cuda')
                pf = M.fuse(p_tab, p_cnn)
                final_pred = M.gate_stage5(M.redecide_23(M.to_label(pf), p_tab23), ev)   # test.py와 같은 최종 예측
                s.update(cnn=macro_f1(ye, M.to_label(p_cnn)), fused=macro_f1(ye, M.to_label(pf)),
                         fused_per_class=per_class(ye, M.to_label(pf)),
                         final=macro_f1(ye, final_pred), final_per_class=per_class(ye, final_pred))
            scores[name] = s
            print(f'  Validation({name}, {len(ev)}행): ' + ', '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in s.items()), flush=True)
        if cnn is not None:
            del cnn
            torch.cuda.empty_cache()

    # ---------------- (2) 최종 학습: 97개 실험 전체 ----------------
    final = M.build_model().fit(Xp, y)
    er = M.pressure_ranges(pool)
    table = M.range_table(er)                                               # 구성별 압력 범위 중앙값 (추론 때 쓰는 유일한 학습 통계)
    Qp = M.pressure_features(pool, table, own_range=M.loo_ranges(er))       # 학습 행은 '자기 실험을 뺀' 추정값으로 계산
    final23 = M.build_model().fit(pd.concat([Xp, Qp], axis=1)[columns23], y)
    bundle = {'model': final, 'columns': columns, 'model23': final23, 'columns23': columns23, 'range_table': table,
              'tab_cfg': M.TAB_CFG, 'lgb_params': M.LGB_PARAMS, 'fuse_w': M.FUSE_W,
              'train_rows': len(pool), 'train_experiments': int(pool.experiment_id.nunique()), 'validation_scores': scores}
    joblib.dump(bundle, args.weights_dir / 'tab_lgbm.joblib', compress=3)
    print(f'저장: {args.weights_dir / "tab_lgbm.joblib"} (기본 모델 피처 {len(columns)}개, 2·3단계 재판정 모델 피처 {len(columns23)}개)', flush=True)

    # ---------------- (3) CNN 가중치 ----------------
    dst = args.weights_dir / 'cnn_pairdiff.pt'
    if args.train_cnn:
        cnn = train_cnn(pool, images, np.ones(len(pool), bool), seed=999)
        save_cnn_fp16(cnn, dst)
        print(f'CNN 새로 학습 → {dst} (GPU 학습은 비트 재현이 안 되어 제출 가중치와 조금 다를 수 있음)', flush=True)
    elif args.cnn_source.exists() and args.cnn_source.resolve() != dst.resolve():
        shutil.copyfile(args.cnn_source, dst)
        print(f'CNN 가중치 복사: {args.cnn_source} → {dst}', flush=True)
    else:
        assert dst.exists(), 'CNN 가중치가 없습니다. --train_cnn 으로 학습하거나 --cnn_source 를 지정하십시오.'
        print(f'CNN 가중치: 기존 {dst} 유지', flush=True)
    print(f'완료 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
