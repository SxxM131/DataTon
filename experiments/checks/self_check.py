"""제출 자가 점검 하니스 (운영진 check_submission.py 6개 항목을 서버에서 직접 재현).

운영진 원본(check_submission.py)은 드라이브에만 있고 서버에는 없어서 CLAUDE.md의 6개 항목을 그대로 구현했다.
  (1) 출력 형식/순서: 열 sample_id,label / 1~6 정수 / 누락·중복·추가 행 없음 / 행 순서 = sample_submission(=입력 순서)
      --final 을 주면 그 CSV와 완전히 같은지도 확인
  (2) 같은 입력 재실행 → 동일
  (3) 입력 행 순서 섞기 → sample_id별 동일
  (4) 실험 나눠 넣기 (--split_mode halves: 실험을 두 묶음으로 / each: 실험 하나씩) → 동일
  (5) 실험별 relative_time_s 앞 60% 행만 → 그 행들의 예측 동일
  (6) 이미지 바꾸기: 센서는 그대로, image_path를 다른 행의 것으로 치환 → 예측이 바뀐 비율 보고 (> 0 이어야 함)
임시 입력/출력은 --work_dir(기본 ~/jaehyeok/cache/seq/self_check_<시각>) 아래에만 쓴다. ~/data 는 읽기만 한다.

예)
  python experiments/checks/self_check.py                                   # 기본: 레포 test.py + 기본 가중치
  python experiments/checks/self_check.py --test_py ../other/test.py --final submission.csv --checks 1,2,5
  python experiments/checks/self_check.py --extra_args "--model weights/m2.joblib" --split_mode each
종료 코드: 모든 점검 통과 0, 하나라도 실패 1.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / 'data' / 'battery_data_light'


class Runner:
    def __init__(self, args):
        self.args, self.n = args, 0
        self.image_root = args.image_root or (args.data_dir / 'test')

    def __call__(self, frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        """frame을 CSV로 저장 → test.py 실행 → 출력 CSV를 읽어 돌려준다."""
        self.n += 1
        inp = self.args.work_dir / f'{self.n:02d}_{tag}_input.csv'
        out = self.args.work_dir / f'{self.n:02d}_{tag}_output.csv'
        frame.to_csv(inp, index=False)
        cmd = [self.args.python, str(self.args.test_py), '--input', str(inp), '--image_root', str(self.image_root),
               '--output', str(out)] + shlex.split(self.args.extra_args)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=self.args.test_py.parent, capture_output=True, text=True)
        (self.args.work_dir / f'{self.n:02d}_{tag}.log').write_text(r.stdout + '\n--- stderr ---\n' + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f'[{tag}] test.py 실패 (code {r.returncode}):\n{r.stderr[-3000:]}')
        print(f'  - 실행 [{tag}] {len(frame)}행, {time.time() - t0:.1f}s', flush=True)
        return pd.read_csv(out, dtype={'sample_id': 'string'})


def same_by_id(ref: pd.DataFrame, new: pd.DataFrame, ids=None):
    """sample_id 기준으로 label 비교. (일치 여부, 불일치 행 수, 비교 행 수)"""
    a = ref.set_index('sample_id')['label']
    b = new.set_index('sample_id')['label']
    ids = list(b.index) if ids is None else list(ids)
    if not set(ids) <= set(b.index) or not set(ids) <= set(a.index):
        return False, -1, len(ids)
    d = int((a.loc[ids].to_numpy() != b.loc[ids].to_numpy()).sum())
    return d == 0, d, len(ids)


def first_frac(frame: pd.DataFrame, frac: float) -> pd.DataFrame:
    """실험별 relative_time_s 오름차순 앞 frac 비율의 행 (최소 1행). 입력 순서는 유지."""
    t = pd.to_numeric(frame['relative_time_s'])
    keep = np.zeros(len(frame), dtype=bool)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        o = idx[np.argsort(t.to_numpy()[idx], kind='mergesort')]
        keep[o[:max(1, int(np.floor(frac * len(o))))]] = True
    return frame[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_py', type=Path, default=REPO / 'test.py')
    ap.add_argument('--data_dir', type=Path, default=DATA)
    ap.add_argument('--input', type=Path, default=None, help='기본 <data_dir>/test/test.csv')
    ap.add_argument('--image_root', type=Path, default=None, help='기본 <data_dir>/test')
    ap.add_argument('--sample_submission', type=Path, default=None, help='기본 <data_dir>/sample_submission.csv')
    ap.add_argument('--final', type=Path, default=None, help='DAKER에 낼 최종 CSV (주면 (1)에서 완전 일치 확인)')
    ap.add_argument('--extra_args', default='', help='test.py에 추가로 넘길 인자 (예: "--model weights/x.joblib")')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--checks', default='1,2,3,4,5,6')
    ap.add_argument('--split_mode', choices=['halves', 'each'], default='halves')
    ap.add_argument('--frac', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--work_dir', type=Path, default=None)
    args = ap.parse_args()
    args.test_py = args.test_py.resolve()
    args.input = args.input or args.data_dir / 'test' / 'test.csv'
    args.sample_submission = args.sample_submission or args.data_dir / 'sample_submission.csv'
    args.work_dir = (args.work_dir or Path.home() / 'jaehyeok' / 'cache' / 'seq' / f"self_check_{datetime.now():%m%d_%H%M%S}").resolve()
    assert not str(args.work_dir).startswith(str((Path.home() / 'data').resolve())), '~/data 아래에는 쓰지 않는다'
    args.work_dir.mkdir(parents=True, exist_ok=True)
    checks = {int(c) for c in args.checks.split(',') if c}
    print(f'test.py: {args.test_py}\n입력: {args.input}\n작업 폴더: {args.work_dir}', flush=True)

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'}, keep_default_na=False)
    # keep_default_na=False: 빈칸을 그대로 다시 써서 원본 CSV와 같은 문자열이 되게 한다 (원본 그대로 보존)
    frame_raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(args.seed)
    run = Runner(args)
    results = []

    def report(no, name, ok, msg):
        results.append((no, name, ok, msg))
        print(f"[{'PASS' if ok else 'FAIL'}] ({no}) {name}: {msg}", flush=True)

    base = run(frame_raw, 'base')

    if 1 in checks:
        sub = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})
        errs = []
        if list(base.columns) != ['sample_id', 'label']:
            errs.append(f'열 {list(base.columns)}')
        if len(base) != len(sub):
            errs.append(f'행 수 {len(base)} != {len(sub)}')
        if base['sample_id'].duplicated().any():
            errs.append('중복 sample_id')
        if base.isna().any().any():
            errs.append('결측')
        if set(base.sample_id) != set(sub.sample_id):
            errs.append('ID 집합 불일치')
        elif not (base.sample_id.to_numpy() == sub.sample_id.to_numpy()).all():
            errs.append('행 순서가 sample_submission과 다름')
        if not (base.sample_id.to_numpy() == frame_raw.sample_id.to_numpy()).all():
            errs.append('행 순서가 입력과 다름')
        lab = pd.to_numeric(base['label'], errors='coerce')
        if not (lab.notna().all() and (lab == lab.round()).all() and lab.isin(range(1, 7)).all()):
            errs.append('label이 1~6 정수가 아님')
        if not str(base['label'].dtype).startswith('int'):
            errs.append(f'label dtype {base.label.dtype} (CSV 안에 소수점 표기?)')
        if args.final is not None:
            fin = pd.read_csv(args.final, dtype={'sample_id': 'string'})
            if not (fin.shape == base.shape and (fin.columns == base.columns).all() and (fin.values == base.values).all()):
                errs.append(f'--final({args.final.name})과 다름')
        dist = base['label'].value_counts().sort_index().to_dict()
        report(1, '출력 형식/순서' + (' + 최종 파일 재현' if args.final else ''), not errs,
               '; '.join(errs) if errs else f'{len(base)}행, 라벨 분포 {dist}')

    if 2 in checks:
        ok, d, n = same_by_id(base, run(frame_raw, 'rerun'))
        report(2, '같은 입력 재실행', ok, f'불일치 {d}/{n}')

    if 3 in checks:
        shuf = frame_raw.iloc[rng.permutation(len(frame_raw))]
        out = run(shuf, 'shuffled')
        ok, d, n = same_by_id(base, out)
        order_ok = (out.sample_id.to_numpy() == shuf.sample_id.to_numpy()).all()
        report(3, '행 순서 섞기', ok, f'불일치 {d}/{n}, 출력이 (섞인) 입력 순서를 따름={order_ok}')

    if 4 in checks:
        exps = frame_raw['experiment_id'].unique()
        groups = [[e] for e in exps] if args.split_mode == 'each' else [list(exps[0::2]), list(exps[1::2])]
        tot_d, tot_n, okall = 0, 0, True
        for gi, g in enumerate(groups):
            part = frame_raw[frame_raw.experiment_id.isin(g)]
            ok, d, n = same_by_id(base, run(part, f'split{gi}'))
            okall &= ok; tot_d += max(d, 0); tot_n += n
        report(4, f'실험 나눠 넣기 ({args.split_mode}, {len(groups)}회)', okall, f'불일치 {tot_d}/{tot_n}')

    if 5 in checks:
        part = first_frac(frame, args.frac)
        out = run(frame_raw.loc[part.index], f'first{int(args.frac * 100)}')
        ok, d, n = same_by_id(base, out)
        report(5, f'미래 행 빼기 (실험별 앞 {args.frac:.0%})', ok and len(out) == len(part), f'불일치 {d}/{n}')

    if 6 in checks:
        # 이미지만 다른 행의 것으로 교체 (센서·시간·ID는 그대로). 전체 무작위 순열 → 자기 자신은 다시 섞어 제외
        perm = rng.permutation(len(frame_raw))
        fixed = np.where(perm == np.arange(len(perm)))[0]
        if len(fixed):
            perm[fixed] = np.roll(perm[fixed], 1) if len(fixed) > 1 else perm[(fixed + 1) % len(perm)]
        swap = frame_raw.copy()
        swap['image_path'] = frame_raw['image_path'].to_numpy()[perm]
        out = run(swap, 'imgswap')
        a = base.set_index('sample_id')['label']
        b = out.set_index('sample_id')['label'].reindex(a.index)
        rate = float((a != b).mean())
        # 같은 실험 안에서만 섞는 더 약한 교란도 함께 보고 (참고용)
        swap2 = frame_raw.copy()
        for _, idx in frame_raw.groupby('experiment_id', sort=False).indices.items():
            swap2.iloc[idx, swap2.columns.get_loc('image_path')] = frame_raw['image_path'].to_numpy()[rng.permutation(idx)]
        b2 = run(swap2, 'imgswap_within').set_index('sample_id')['label'].reindex(a.index)
        rate2 = float((a != b2).mean())
        report(6, '이미지 바꾸기', rate > 0,
               f'예측 변경 비율 {rate:.1%} (전체 무작위 교체) / {rate2:.1%} (같은 실험 안에서만 교체, 참고)')

    print('\n요약')
    for no, name, ok, msg in results:
        print(f"  ({no}) {'PASS' if ok else 'FAIL'}  {name}: {msg}")
    print(f'test.py 실행 {run.n}회, 임시 파일: {args.work_dir}')
    sys.exit(0 if all(ok for *_, ok, _ in results) else 1)


if __name__ == '__main__':
    main()
