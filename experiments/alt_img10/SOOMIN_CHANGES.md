# soomin 브랜치 변경 요약

> **이 문서는 대안 실험(이미지 피처 10개 + 융합 0.1)의 기록입니다.** 최종 제출 파이프라인은 `README.md`에 적힌 구성이며, 루트의 `test.py`·`weights/`는 이 문서의 선택안이 아니라 최종 구성을 따릅니다. 아래 본문은 실험 당시(`soomin` 브랜치) 기준으로 쓰여 있습니다. 채택하지 않은 이유: 제공 분할 1회(Validation 19개 실험)에서는 0.9254 → 0.9349였지만, 97개 실험 교차검증(5-fold × 3회)에서는 최종 구성과 차이가 없었습니다 (같은 규칙 보정 기준 0.9226 vs 0.9229).
>
> 이 폴더의 파일은 레포 루트에 있던 것을 내용 그대로 옮긴 것입니다. 본문과 스크립트에 적힌 경로(`metrics.csv`, `tests/`, `logs/soomin/` 등)는 옮기기 전의 루트 기준입니다. 레포에 없는 로컬 파일 `artifacts/`가 필요해 그대로는 실행되지 않습니다. 루트로 복사해 실행하면 `fit_final_img10.py`가 `weights/tab_lgbm.joblib`을 덮어쓰므로, 최종 제출을 재현할 때는 실행하지 마십시오.

작업 공간: `~/soomin/datathon_img/` (원본 `DataTon`·`dataton_final` 폴더는 **수정하지 않음**, 복사본에서만 작업).

기준 비교:

| 기준 | 의미 | Public / 점수 앵커 |
|---|---|---|
| **DataTon** (`main` / jaehyeok v7) | 리더보드 **0.9497** 파이프라인 | CV 0.9212, 공식 test_like final ≈0.9263 |
| **dataton_final** (`~/soomin/dataton_final`) | v5 계열 (gate5/6만, hold·floor 없음) | 공식 test_like final **0.8984** |

평가·선택 숫자는 `metrics.csv`, `artifacts/selection.json`(로컬)에서 확인.

---

## 1. dataton_final → DataTon(v7) 에서 이미 있던 차이 (원본 기준, soomin이 만든 것 아님)

`train.py`는 동일. `model.py` / `test.py` 후처리만 확장됨.

| 항목 | dataton_final (v5) | DataTon v7 |
|---|---|---|
| `gate_stage5` | `out == 5` 만 4로 | `out >= 5` (6도 근거 없으면 4) |
| `still_climbing` / `gate_stage6` | running-max **+** `PEAK_RATE=0.2℃/s` | running-max만 (속도 조건 제거) |
| `floor_stage4` | 없음 | 원통형 4단계 최소 **30초** |
| `hold_stage` | 없음 | 4·5 단조 유지 (`vdrop_seen` 후 4 굳힘) |
| `fuse_w` | 0.2 | 0.2 |
| 제출 | `sub14` ≡ dataton_final | `sub16` Public **0.9497** |

soomin 작업은 **이 v7 게이트를 유지**한 채 이미지 결합만 바꿨다.

---

## 2. soomin이 DataTon(v7) 대비 수정·추가한 것

### 2.1 `model.py`

**추가** (기존 함수는 유지):

- `IMG10_BASE`, `is_image_col()`, `color_delta10()`, `build_features_img10()`
- 역할: 기존 `pt_*` 윈도우 다수를 쓰지 않고, **크기 불변 비율 5개 + 직전 프레임 대비 변화 5개(총 10)** 만 센서·규칙 피처와 결합

**유지**: v7 `apply_rules` 체인  
`redecide_23 → gate6 → gate5 → floor_stage4 → hold_stage`

### 2.2 `test.py`

| 항목 | DataTon v7 | soomin |
|---|---|---|
| 표 피처 | `build_features` (296열, 색 윈도우 포함) | `feature_mode=='img10'` 이면 `build_features_img10` |
| 융합 가중 | 고정 docstring상 0.2 | `bundle['fuse_w']` (선택안 **0.1**) |
| 환경 | `OMP_NUM_THREADS=6` | `OMP=2`, `CUDA_VISIBLE_DEVICES=0` |
| CNN | `weights/cnn_pairdiff.pt` | **동일 파일 재사용 (재학습 없음)** |

### 2.3 `train.py`

**변경 없음** (md5 동일). 최종 img10 가중치는 `fit_final_img10.py`로 별도 fit.

### 2.4 신규 파일

| 파일 | 용도 |
|---|---|
| `run_img_combine.py` | train-only 표 + train-only CNN으로 test_like 후보 A/B/C 비교 |
| `fit_final_img10.py` | 선택안 C를 Train+Val 97실험로 fit → `weights/tab_lgbm.joblib` + 제출 CSV |
| `metrics.csv` | 후보별 test_like Macro F1, Δ, 클래스별 F1, 빈 이미지 변화율 |
| `SOOMIN_CHANGES.md` | 본 문서 |
| `tests/check_repro.py` | 행 섞기·실험 분리·미래 절단·빈 이미지 |
| `tests/check_robust.py` | 소량 스모크 |
| `submissions/img10_fuse01_v7gates.csv` | 선택안 제출 파일 |

### 2.5 `weights/tab_lgbm.joblib`

- **재학습됨** (img10 피처, Train+Val 97실험, `fuse_w=0.1`, `feature_mode=img10`)
- `cnn_pairdiff.pt`는 DataTon과 **동일** (신규 CNN 학습 없음)

---

## 3. 실험 결과 요약 (출처: `metrics.csv`)

평가: validation **test_like** 6514행, **train-only** LightGBM + **train-only CNN** 확률, v7 게이트 적용.

| 후보 | test_like F1 | Δ vs 표+게이트 | 빈 이미지 변화율 |
|---|---:|---:|---:|
| baseline 표+게이트 | 0.925409 | 0 | 5.80% (색 재계산) / **0%** (캐시 색 유지) |
| A `fuse_w=0.2` | 0.926284 | +0.000875 | 5.56% |
| B `w=0.2,m=0.3` | 0.926452 | +0.001043 | 5.56% |
| **C img10 + fuse 0.1 (선택)** | **0.934938** | **+0.009528** | **1.37%** |

선택 기준: 빈 이미지 변화 ≥1% 이면서 test_like 손실 최소(실제로는 이득 최대) → **C_img10_fuse_w0.1**.

CNN 확률: `val_testlike_proba_trainonly_cnn.joblib` (full `cnn_pairdiff.pt`와 가중치 상이 확인).

---

## 4. 재현·규정 점검 (제출 CSV 기준)

| 검사 | 결과 | 출처 |
|---|---|---|
| check_repro 4종 | 통과 (빈 이미지 **0.55%** 변경) | `logs/check_repro.json` |
| check_robust | 통과 | `logs/check_robust.json` |
| check_submission | 모두 통과 | `logs/check_submission.log` |

---

## 5. 실행 방법

```bash
# 추론
python test.py \
  --input <battery_data_light>/test/test.csv \
  --image_root <battery_data_light>/test \
  --output submissions/img10_fuse01_v7gates.csv \
  --sample_submission <battery_data_light>/sample_submission.csv \
  --device cuda

# (선택) 후보 재비교 — 로컬 artifacts 필요
python run_img_combine.py

# (선택) img10 가중치 재학습 후 제출 재생성
python fit_final_img10.py
```

---

## 6. 의도적으로 건드리지 않은 것

- `~/jaehyeok/DataTon`, `~/choiajin/DataTon`, `~/soomin/dataton_final` 원본 트리
- CNN 재학습
- v7 게이트 상수·순서 (`STAGE4_MIN_S`, surge/vdd cfg 등)
