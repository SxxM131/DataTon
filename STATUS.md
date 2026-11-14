# STATUS — 진행 상황 인수인계 (jaehyeok 트랙)

> 새 세션은 `CLAUDE.md`(규칙) → 이 파일(현황) 순서로 읽고 이어서 작업한다.
> 최종 갱신: 2026-11-14 08:05 UTC (17:05 KST). 리더보드 마감 11/15 13:00 KST, 메일 제출 13:30 KST.

## 작업 규칙 (사용자 지시)

- Claude는 사령탑 역할. 일은 하위 세션(subagent)에 나눠 맡기고, **하위 세션을 만들기 전에 어떤 모델을 쓸지 먼저 보고**한다.
- **git commit / push는 먼저 제안하고 허락을 받은 뒤** 진행. 브랜치는 `jaehyeok`.
- 서버에 GitHub 인증이 없다 → **push는 사용자가 터미널에서 직접** (`git push origin jaehyeok`). 토큰을 서버에 저장하지 않는다.
- **리더보드 제출은 사용자에게 부탁**한다 (파일 경로 + 낼 가치가 있는 이유를 같이).
- 서버는 팀 공유 계정. 다른 팀원 작업(`~/soomin` 등)이 CPU 18코어를 자주 다 쓴다 → 항상 `OMP_NUM_THREADS` 4~8, `n_jobs` ≤ 6. 남의 프로세스는 건드리지 않는다.
- 공용 venv에 `lightgbm 4.7.0`, `timm 1.0.31` 설치함 (사용자 승인). sklearn `HistGradientBoosting`은 CPU 경합 시 극단적으로 느려지므로 LightGBM을 쓴다.

## 지금 점수

| 모델 | 공통 CV (90% 구간) | 긴 실험 / 짧은 실험 | 제공 분할(Test식 변환) | Public LB |
|---|---|---|---|---|
| 베이스라인 v7 (LR, 센서 5 + 그레이스케일) | 0.810 (0.781~0.834) | 0.637 / 0.832 | 0.746 | **0.8517** (sub01) |
| 기준 GBDT `ref_gbdt_color_extra` (HGB, 센서+30초 윈도우+색 통계+추가 컬럼) | 0.886 (0.859~0.908) | 0.724 / 0.901 | 0.846 | 미제출 (sub02 준비됨) |
| **`tab_all_pool`** (LightGBM, 다중 윈도우·누적 피처 323개) | **0.895** (0.874~0.913) | **0.820** / 0.898 | 0.894 | 미제출 |

- 클래스별 F1 (`tab_all_pool`): [1.00, 0.77, 0.89, 0.91, 0.92, 0.89] → 지금 약점은 **2단계 vs 3단계**.
- `tab_all_pool` vs 기준: +0.016, 개선일 확률 0.95 (채택 기준 통과).
- 베이스라인 기준으로 Public이 CV보다 0.04 높다. Test 크기(실험 11개) 점수의 표준편차가 약 0.04라서 오차 범위 안이다.

## 검증 체계 (반드시 이걸로 비교) — `experiments/common/cv.py`

**왜 바꿨나**: Test는 500행 넘는 실험을 500행으로 균등 다운샘플해 놓았다(`floor(linspace(0, N-1, 500))` 규칙이 Test의 10/15초 간격 패턴과 정확히 일치). Train/Validation은 원본 5초 간격 그대로라, 긴 실험이 행의 53%(Train)·66%(Validation)를 차지한다. Test에서는 29%다. 원본 행 그대로 잰 Validation 점수(베이스라인 0.678, GBDT 0.788)는 Test와 가중치가 달라서 믿을 수 없다. **이전 방식의 점수는 폐기.**

- `pool = cv.load_pool(data_dir)`: Train+Validation 97개 실험 전체.
- `evalp = cv.to_test_like(pool)`: 채점 행. Test와 같은 500행 상한. **시계열 피처도 evalp 행만 보고 만든다.**
- `cv.make_folds(pool)`: 실험 단위 층화(bat_class × tr_method) 5-fold × 3회. 시드 고정.
- `cv.evaluate` → CV, 클래스별, 긴/짧은 실험, bat_class별, 실험 단위 부트스트랩 구간, Test 크기 모의.
- `cv.compare(evalp, oof_a, oof_b)`: 실험 단위 paired bootstrap. **`p_a_better ≥ 0.9`일 때만 개선으로 인정.**
- `cv.official_split(pool)`: 제공 분할(Train 78 → Validation 19) 점수. README용 보조 지표.
- 결과 저장: `experiments/results/<이름>.json` (커밋 대상), `experiments/results/oof/<이름>.pkl` (커밋 금지, .gitignore).
- 최종 파일은 Public 최고점이 아니라 **CV 기준으로 고른다.** Public과 Private 차이의 표준편차는 0.015로 추정되지만, 이는 행 단위 무작위 30% 가정이다. 실험 단위로 나뉘었다면 훨씬 크다.

## 확인된 데이터 사실 (EDA)

- 라벨은 97개 실험 모두에서 **단조 증가** (1→2→3→4→5→6, 일부는 4나 5를 건너뜀). 단계별 지속 시간 중앙값 330 / 210 / 478 / 70 / 80 / 70초.
- CSV에는 `CLAUDE.md` 표에 없는 컬럼이 있다: 가스 10종(`meas_ch4`~`meas_so2`), `atmp_t`, `thermal_t`, `indu_t`, `indu_t.1`, 메타(`bat_class`, `bat_cc`, `soc_status`, `tr_method`), `meas_t1_plus_t2`.
  - `meas_a`는 전부 0. `meas_t1_plus_t2`는 t1+t2. **`indu_t.1`은 Test에서 100% 결측 → 사용 금지.**
  - 추가 컬럼은 써도 된다고 팀에서 판단(Test에도 있고 금지 규칙에 해당 없음). 다만 **"센서 5개 + 이미지만" 버전 점수를 항상 같이 기록**해 둔다.
- 열화상은 **무지개 의사색상**: 마젠타(약 295°, 배경·최저온) → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색(최고온). 그레이스케일 밝기는 온도와 단조 관계가 아니다. 의사 온도 `pt = clip((300 − hue°)/300, 0, 1)`, 흰색은 1 (`exp01_image_color.color_stats`).
- 이미지 크기는 실험마다 다르다 (37×76 ~ 336×297, 대형 셀 300px대 / 소형 셀 폭 40~60px). 한 실험 안에서는 거의 같다.
- Test 11개 실험 중 2개가 500행으로 축소된 긴 실험(10/15초 간격). 윈도우는 반드시 초 단위.

## 트랙별 진행 상황

하위 세션 3개(A·C는 Opus 5.5, B는 Fable 5.2)가 **사용량 한도에 걸려 약 10분 만에 중단**됐다. 세션 문맥은 사라지지만 코드와 캐시는 남아 있다. 각 폴더에 NOTES.md는 아직 없다.

### A. 센서·GBDT — `experiments/tab/`
- `feats.py`: 벡터화된 피처 생성 `build(frame, color, cfg)`. 다중 윈도우(30/120/300초), 누적 최댓값·최솟값, 경과 시간류. Test 11개 실험 피처 생성 43ms.
- `run.py`: 공통 검증 러너. 옵션 `--feat sensor5|all`, `--train_rows pool|tl|pool+tl|pool+tl3`, `--weight none|exp|sqrt|class|exp+class`, `--cfg`, `--seeds`.
- 완료: `tab_all_pool` 1건 (위 표). 중요 피처는 `meas_t1_cummax`, `meas_t1_max300s`, `meas_v_cummin` 등 누적·긴 윈도우.
- **남은 일**: ① `--feat sensor5` 버전 채점 ② 긴 실험 약점 원인 진단(학습에 Test식 다운샘플 복사본 추가 `pool+tl`, 실험 균형 가중치) ③ 2 vs 3단계 개선 ④ 피처 가지치기(323개는 많다) ⑤ 확률 저장해서 C에 전달.

### B. 이미지 — `experiments/img/` (GPU)
- 캐시: `~/jaehyeok/cache/img/pool_rgb128.npy`, `pool_pt128.npy` (97개 실험 이미지 128px), `handfeat.pkl`(수작업 공간 피처 28개), `cnn_temporal.npz`(CNN 확률), `cnn_temporal_out*.pt`(가중치 각 22MB).
- 결과 1, **수작업 이미지 피처는 효과 없음**: 센서만 0.881 / 센서+색 통계 0.881 (차이 −0.001, 개선 확률 0.38) / 센서+색+수작업 28개 0.868 (**악화**, 개선 확률 0.01). 수작업 이미지 피처만으로는 0.745.
- 결과 2, **CNN은 이미지만으로 0.67~0.80**: `resnet18.tv_in1k`(ImageNet, BSD-3-Clause) 미세조정, 입력은 현재 + 약 30초 전 + 실험 첫 프레임(9채널). fold별 0.737 / 0.799 / 0.667 / 0.749 / 0.795, 제공 분할 0.725. fold당 학습 약 20초.
- **남은 일**: `stack.py`로 CNN 확률을 센서 GBDT에 스태킹했을 때 유의한 개선이 있는지 (누수 없는 중첩 구조 `pair k,j`는 이미 학습됨). 이게 "이미지가 실제로 기여한다"를 보일 핵심 실험이다. 사전학습 임베딩(`embed.py`)은 미실행.

### C. 순서 보정·제출 검증 — `experiments/seq/`, `experiments/checks/`
- `checks/self_check.py`: **운영진 6개 점검 자체 검사 완성.** 베이스라인 `test.py`는 6개 모두 통과. 이미지 교체 시 예측 변경 비율 15.6% (같은 실험 안에서만 교체하면 11.2%).
  - `python experiments/checks/self_check.py [--test_py ...] [--final <csv>] [--extra_args "..."] [--checks 1,2,5]`
- `seq/make_proba.py`: 기준 피처로 OOF 확률 생성 완료 → `~/jaehyeok/cache/seq/proba_inner.pkl`.
- `seq/decode.py`: `fit_params(label, time_s, exp_id)`, `decode(proba, time_s, exp_id, params, method='hmm', ...)` 작성됨. **아직 채점 안 함.**
- **남은 일**: argmax / 단조 고정 / HMM 전방 필터 비교, 60% 잘라 넣기·행 섞기 인과성 검증, A가 넘겨줄 확률로 재평가.

## 최종 제출까지 남은 일

1. A·B·C 남은 일 마무리 → 가장 좋은 조합 확정 (테이블 모델 + CNN 확률 스태킹 + 인과적 순서 보정).
2. `model.py / train.py / test.py` 3파일로 통합. CNN 가중치 포함 `weights/` 100MB 이하 목표. `test.py`는 학습·fit 없이 결정적으로.
3. `self_check.py`로 6개 점검 통과 확인. 드라이브의 공식 `check_submission.py`는 서버에 없다 → 팀원에게 받아서 한 번 더.
4. 최종 학습은 Train+Validation 97개 실험 전체로 (README에 명시).
5. README의 `<>` 칸 채우기, 사전학습 모델 출처·라이선스 기재.

## 제출 기록 (11/14: 1/30회 사용)

| 파일 (`submissions/`, 레포에 안 올라감) | 내용 | 생성 방법 | CV | Public |
|---|---|---|---|---|
| `sub01_baseline_v7.csv` | 베이스라인 v7 | `test.py` + `weights/model.joblib` | 0.810 | 0.8517 |
| `sub02_ref_gbdt.csv` | 기준 GBDT, 97개 실험 전체 학습 | `experiments/common/predict_ref_gbdt.py` | 0.886 | **미제출** |

- `sub02`는 CV가 0.076 오를 때 Public이 얼마나 따라오는지 보는 보정용이다. 최종 제출용이 아니다(최종은 반드시 `test.py`로 생성).

## 파일 위치

| 무엇 | 어디 |
|---|---|
| 공통 검증 | `experiments/common/cv.py`, `ref_gbdt.py`, `ref_baseline.py`, `predict_ref_gbdt.py` |
| 초기 실험 (이전 검증 방식, 참고용) | `experiments/exp01_image_color.py`, `exp02_gbdt.py`, `logs/` |
| 결과 JSON | `experiments/results/*.json` |
| 캐시 (레포 밖, 3GB) | `~/jaehyeok/cache/` — 색 통계 `{train,validation,test}_color.pkl`, 그레이 `*_gray.pkl`, `img/`, `seq/`, `tab/` |
| 팀 공유용 EDA 문서 | `~/jaehyeok/eda_share/` — `EDA_notion.md`, `eda.html` (레포 밖. 예시 실험 원본 시계열이 들어 있어 공개 링크는 레포에 적지 않는다) |

## Git 상태

- `jaehyeok` 브랜치, 커밋 `63539e4`(exp01·exp02·로그)가 **아직 push 안 됨**.
- 커밋 안 된 것: `STATUS.md`, `CLAUDE.md` 수정, `.gitignore` 수정, `experiments/{common,tab,img,seq,checks,results}/`.
- 올리지 않는 것: `weights/model.joblib`(베이스라인 가중치), 루트 `train.log`, `submissions/`, OOF 예측.
