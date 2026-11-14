# 사실표 (사실 | 수치 | 출처)

최종 모델 **C** = DataTon 표 모델 + 이미지 피처 10개(img10) + CNN 후기 융합 w=0.1 + v7 게이트.  
제출 CSV: `~/soomin/datathon_img/submissions/img10_fuse01_v7gates.csv` (md5 `8660d1194b87fb69f38d434e8fb2d51f`).

---

## 1. 데이터

| 사실 | 수치 | 출처 |
|---|---|---|
| train 규모 | 33,522행 / 78실험 | `eda/EDA_SUMMARY.md` §1 |
| validation 규모 | 10,200행 / 19실험 | 同上 |
| test 규모 | 3,451행 / 11실험 | 同上 |
| 실험당 행 중앙값 (min–max) | train 290(147–1,212); val 302(97–1,543); test 295(222–500) | 同上 |
| 라벨 단조성 | train+val 97실험 모두 1에서 시작·단조 증가(감소 0) | `eda/EDA_SUMMARY.md` §2; `datathon_img/STATUS.md` EDA |
| 단계 건너뛰기 | train 8 / val 3실험 (4·5·6 중 하나) | `eda/EDA_SUMMARY.md` §2 |
| 1–6 전부 실험 | train 67/78(85.9%); val 15/19(78.9%) | 同上 |
| 단계 길이 중앙값(초, train 비절단) | 1:330 / 2:210 / 3:478 / 4:70 / 5:82; 6 관측 65(절단) | 同上 |
| 샘플링 간격 train·val | 전 실험 5초 | 同上 §1 |
| test 간격 | 9실험(행 71%) 5초; 2실험(행 29%) 10·15초 혼합, 각 500행 캡 | 同上 |
| 이미지 크기 최다 | 336×297 = 전체 30.7% | 同上 §4 |
| 짧은 변 중앙값 | train 255 / val 297 / test 151 | 同上 |
| 짧은 변 100 미만 | 전체 20.8%; test 38.2% | 同上 |
| meas_t1–라벨 순위상관 | train 0.92 / val 0.92 | 同上 §3 |
| meas_t1 라벨별 중앙값 방향 | 라벨1 ≈34 → 라벨6 ≈300 | 同上 |

---

## 2. 최종 모델 구조 (C)

| 사실 | 수치 | 출처 |
|---|---|---|
| 표 피처 모드 | `feature_mode=img10` (non_img + img10 10열; joblib columns≈235) | `DataTon_github/SOOMIN_CHANGES.md`; `logs/redecide23_baseline_spec.md` |
| img10 피처 10개 | `img10_pt_frac50/70/90`, `img10_pt_p90/p99`, `img10_d_pt_frac50/70/90`, `img10_d_pt_p90/p99` | `artifacts/selection.json` |
| 표 모델 | LightGBM multiclass (+ 2·3 재판정용 model23) | `datathon_img/README.md` / DataTon README |
| CNN | pairdiff ResNet18, 9ch(현재+직전+차이), 128×128; 가중치 `cnn_pairdiff.pt` 재사용(재학습 없음) | `SOOMIN_CHANGES.md` §2.2 |
| 후기 융합 | \(P \propto P_{tab}^{0.9}\cdot P_{cnn}^{0.1}\) (`fuse_w=0.1`) | `selection.json`; `submit.log` |
| v7 게이트 체인 | `redecide_23 → gate6 → gate5 → floor_stage4 → hold_stage` | `SOOMIN_CHANGES.md` §2.1 |
| 최종 학습 범위 | Train+Val 97실험 (`fit_final_img10.py`) | `SOOMIN_CHANGES.md` §2.5 |
| CSV md5 | `8660d1194b87fb69f38d434e8fb2d51f` | `artifacts/baseline_C_final_20261114_145918/md5.txt` |
| Public LB (C) | **0.9540** | **[확인 필요]** — 로컬 파일에 미기록, 사용자 지시값 |

---

## 3. 리더보드 이력

| 단계 | 수치 | 출처 |
|---|---|---|
| hgb softmono | LB **0.8597** | `NOTES.md` (ens 리더보드 결정) |
| ens (hgb+xgb) RM | LB **0.8473825147** (≈0.8474) | 同上 |
| xgb RM | LB **0.8245** | **[확인 필요]** — 사용자 지시; 로컬에 Public 기록 미발견 |
| dataton_final (v5) | Public **0.9375** (CV≈0.9146, test_like final 0.8984). 사용자 표기 “0.92”는 근사 표현 | `datathon_review/review.md`; `STATUS` sub14 |
| DataTon 표+게이트 v7 | Public **0.9497** | `SOOMIN_CHANGES.md`; `STATUS` sub16 |
| C (img10+fuse0.1+v7) | Public **0.9540** | **[확인 필요]** 사용자 지시 |

---

## 4. 이미지 근거

| 사실 | 수치 | 출처 |
|---|---|---|
| C ablation vs 표+게이트 | test_like Δ **+0.009528** (0.925409 → 0.934938) | `metrics.csv` / `selection.json` |
| 빈 이미지 변화율 (val test_like, C) | **1.366%** (0.01366288) | `metrics.csv` `blank_change_rate` |
| 빈 이미지 변화율 (test, check_repro) | **0.5506%** (0.00550565); check_submission 0.55% | `logs/check_repro.json`; `selection.json` |
| 전이 정렬 — 이미지 선행 | **4→5·5→6**에서 가장 뚜렷 (`img_hotspot_move` earlyΔ≈3.97 / 3.83; `white_frac` lead_ratio>1.9) | `artifacts/transition_lead_scores.csv`; `logs/advance_report.json` |
| 의사온도 | `pt = clip((300−hue°)/300, 0, 1)`, 흰색=1 | `STATUS.md` EDA / color_stats |

---

## 5. 검증

| 사실 | 수치 | 출처 |
|---|---|---|
| test_like 정의 | 실험당 500행 상한 균등 다운샘플; 피처도 그 행만으로 생성 (Test와 동일 조건) | `STATUS.md` 검증 체계; `SOOMIN_CHANGES.md` |
| C 선택 시 test_like n | **6,514**행 (train-only 표+train-only CNN) | `selection.json` |
| test_like vs LB 순위 | 표+게이트(0.925) < C(0.935) 이고 LB도 v7(0.9497) < C(0.9540)로 **같은 방향** | `metrics.csv` + LB 이력 (C LB는 확인 필요) |
| C OOF (CNN 생략 근사) | Macro F1 **0.924359** (n=31,784) | `artifacts/oof_C/meta.json`; `logs/run_oof_tau_gate56.log` |
| OOF 근사 vs 실제 C val | approx_val(no CNN) 0.931983; C actual 0.934938; 차이 −0.00295 | `oof_C/meta.json` |
| τ only OOF | base 0.924359 → 0.923921 (Δ **−0.000438**) → **기각** | `logs/oof_tau_gate56_report.json` |
| gate56 only OOF | → 0.920712 (Δ **−0.003648**) → **기각** | 同上 |
| τ+gate56 OOF | → 0.921231 (Δ **−0.003129**) → **기각** (`REJECT keep C`) | 同上; `logs/oof_tau_gate56_decision.txt` |
| C test_like 클래스별 F1 | 1:1.000 / 2:0.859 / 3:0.951 / 4:0.947 / 5:0.950 / 6:0.903 | `metrics.csv` C_img10_fuse_w0.1 |
| 이웃 오분류 (C test_like) | 2↔3:190 / 3↔4:6 / 4↔5:30 / 5↔6:40 | `metrics_advance.csv` |

---

## 6. 기각한 시도

| 시도 | 수치 | 기각 이유 | 출처 |
|---|---|---|---|
| HMM (xgb 확률) | OOF HMM 0.8789 < softmono 0.8936 / RM 0.8886; val HMM 0.8378 ≪ RM 0.9025 | RM/softmono보다 열등, `adopt_hmm=false` | `experiments/hmm_forward/metrics.json` |
| R1·R2 규칙 피처 (xgb_v2) | OOF C(+rules+const) 0.8959 vs A 0.8872 (+0.007 < +0.01); val 0.8787 < 기준 0.9025 | RULES +0.01 미달·val 악화 | `NOTES.md` xgb_v2_rules |
| 압력 피처 B/C | OOF RM A 0.8886 / B 0.8864 / C 0.8850; val 악화 | 이득 없음·단순 A 유지 | `experiments/pressure_ab/metrics.json` |
| D (C+신규 이미지 전이 피처) | test_like Δ **−0.0100**; blank 1.84% | +0.005 채택선 미달 | `metrics_advance.csv` |
| E (img10 relative) | Δ **−0.0074**; blank 0.54%(기준선↓) | 채택선 미달·blank 악화 | 同上 |
| 2·3 재판정 압력 추가 | OOF Δ **−0.001226** (0.92643→0.92521) | 개선 없음 | `metrics_redecide23.csv` |
| E_e5bc 특이 보정 | 해당 실험만 4스킵·오류 대량; “E_e5bc 단독 패턴은 보정 금지” | 단일 실험 과적합 | `experiments/dataton_err/REPEAT_PATTERNS.md` |
| A/B 융합(원본 색 윈도우) | A w0.2 Δ+0.0009; B best Δ+0.0010 | blank≥1% 중 C가 이득 최대 | `metrics.csv` |

---

## 7. 규정·재현

| 사실 | 수치 | 출처 |
|---|---|---|
| check_repro 4종 | shuffle / 실험분리 / 미래절단 / 빈이미지 → **all_pass=true** | `logs/check_repro.json` |
| 미래절단 동일 행 수 | truncate_n=2,763 (남은 행 예측 동일) | 同上 |
| check_robust | all_ok=true | `logs/check_robust.json` |
| check_submission | 모두 통과; 검은 이미지 0.55% | `logs/check_submission.log`; `selection.json` |
| 이미지 누락 처리 (soomin 공통) | 없거나 실패 → 같은 실험 **과거** 이미지 피처 carry-forward, 없으면 NaN | `NOTES.md` 2026-11-14 |
| C test.py 이미지 | 파일에서 직접 `extract_image_features` (캐시 미사용) | `logs/advance_report.json` cache step |

---

## 8. 한계 (인과적 상한)

| 사실 | 수치 | 출처 |
|---|---|---|
| 3단계 문서 정의 | 50–90℃ & **+20초** 압력차 0.02 (실험 전체 정규화) — **미래·전체통계** | `STATUS.md` 가이드 원문 |
| 비인과 문서규칙 상한 | CV **0.9353** (규정 준수 후보 당시 0.9029) | `STATUS.md` noncausal_doc_rules |
| 과거만 미래창·최고점 피처 상한(초기) | 0.900 → 0.914 → 0.925 | `STATUS.md` noncausal_bound |
