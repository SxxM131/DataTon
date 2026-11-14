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

## 대회 명세서 원문 확인 (11/14 17:40 KST, 사용자가 전달)

- 명세서의 "주요 항목"은 `sample_id`, `image_path`, `meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p`, `label`, `experiment_id`, `relative_time_s` 10개뿐이다. **가스·`thermal_t`·`indu_t`·배터리 메타 등 나머지 컬럼은 설명이 없다** (금지 문구도 없음). 단계 1~6의 정의, 이미지 팔레트 설명도 없다.
- Public/Private 분할 방식은 비공개. "Public Score는 최종 순위를 보장하지 않으며, 검증 데이터의 성능과 함께 모델을 판단"하라고 명시되어 있다.
- **"리더보드 점수로 개별 정답을 역추적하거나 예측값을 임의 조작할 수 없다."** → 진단용 제출은 정상 모델의 예측 그대로만 낸다. 어떤 실험이 Public인지 알아내려는 제출은 하지 않는다.
- 제출값은 정상적인 모델 추론 코드로 생성. 최종 파일 미선택 시 Public 최고점 파일이 자동 선택된다 → **반드시 직접 선택.**
- 메일 제출 마감은 명세서에 13:50으로 적혀 있지만, OT 안내(13:30)를 기준으로 한다.
- 제공 데이터와 정답의 외부 공개·재배포 금지. 생성형 AI는 코딩 보조로 허용(데이터 입력 금지 문구는 `CLAUDE.md`의 OT 구두 안내 참고).

## 지금 점수

| 모델 | 공통 CV (90% 구간) | 긴 실험 / 짧은 실험 | 제공 분할(Test식 변환) | Public LB |
|---|---|---|---|---|
| 베이스라인 v7 (LR, 센서 5 + 그레이스케일) | 0.810 (0.781~0.834) | 0.637 / 0.832 | 0.746 | **0.8517** (sub01) |
| 기준 GBDT `ref_gbdt_color_extra` (HGB, 센서+30초 윈도우+색 통계+추가 컬럼) | 0.886 (0.859~0.908) | 0.724 / 0.901 | 0.846 | 미제출 (sub02 준비됨) |
| `tab_all_pool` (LightGBM, 다중 윈도우·누적 피처 323개) | 0.895 (0.874~0.913) | 0.820 / 0.898 | 0.894 | 미제출 |
| **`tab_all_nocum_r3`** (LightGBM, 누적 피처 제외 266개, 3회 반복) — **현재 최고** | **0.904** (0.884~0.920) | 0.816 / 0.910 | 0.892 | **0.8309** (sub08) |
| `tab_s5_nocum_rule_r3` (센서 5개 + 이미지 색 통계만, 대체안) | 0.900 (0.880~0.916) | 0.820 / 0.904 | 0.878 | 미제출 (sub09 준비됨) |

- 클래스별 F1 (`tab_all_nocum_r3`): [1.00, 0.82, 0.90, 0.90, 0.92, 0.88] → 약점은 여전히 **2단계 vs 3단계**.
- `tab_all_nocum_r3` vs 기준: +0.018, 개선일 확률 0.99. 같은 조건에서 추가 컬럼 포함(all)이 센서 5개만(sensor5)보다 +0.006 (개선 확률 0.98).
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

하위 세션 A(Opus 5.5)·B(Fable 5.2)는 보고를 마치고 종료. C는 사령탑이 직접 진행. **세션 D(라벨 규칙 찾기, Fable 5.2)가 11/14 17:50 KST에 시작되어 진행 중** (`experiments/rules/`, 기록은 `experiments/rules/NOTES.md`).

**라벨은 센서 규칙 기반으로 보인다**: 2단계는 `meas_t1 >= 45`가 처음 되는 행에서 시작(97/97 정확히 일치). 3단계는 `meas_t1 >= 50` + 미확인 조건. 4단계는 전압 급락과 관련(단순 임계값은 31~42%만 정확). 3~6단계 규칙을 세션 D가 찾는 중.

### A. 센서·GBDT — `experiments/tab/` (세션 종료, 보고 완료. 자세한 표는 `experiments/tab/NOTES.md`)
- **추천 모델 `tab_all_nocum_r3`**: `feats.build(frame, color, {'extra': True, 'cum': False})` 266개 피처 + LightGBM(lr 0.1, 200회, 잎 31, colsample 0.3, max_bin 63, deterministic). 학습은 pool 원본 5초 행 그대로, 증강·가중 없음.
  - 피처: 원시 센서·가스·메타·이미지 색 통계, 30/120/300초 윈도우 mean·max·min, 15~300초 전 대비 변화량과 초당 변화율, 임계값 첫 도달 이후 경과초(t1 45~300, t2 100/200, 전압 급락).
  - 피처 생성은 벡터화되어 Test 4k행에 40ms. 행 섞기·실험 1개만·앞 60%만 넣어도 피처가 동일함을 확인.
- **대체안** (센서 5개 + 이미지 색 통계만): `--feat sensor5 --cfg '{"cum":false,"rule":true}'`, CV 0.900.
- **전체 학습 + Test 예측**: `OMP_NUM_THREADS=6 python experiments/tab/fit_predict.py --input <test.csv> --image_root <test 폴더> --output <csv>` (약 5초, 확률은 `_proba.csv`로 같이 저장).
- OOF 확률: `experiments/results/oof/tab_all_nocum_r3.pkl` (`p1_r0`~`p6_r2`).
- 알게 된 것:
  - 누적 최댓값·최솟값 피처는 2단계를 해친다 (bat 2 × tr 2 짧은 실험에서 2단계 F1 0.51). 빼면 CV가 오른다.
  - 긴 실험 약점의 원인은 간격 불일치가 아니다. 긴 실험은 4·5단계가 20~38초라 10/15초 간격에서 0~2행만 남고, 실험 하나(E_e5bc…, 5단계가 3,000초)가 긴 실험 5단계 행의 58%를 차지한다. 긴 실험 점수 차이는 신중히 해석.
  - 학습에 Test식 다운샘플 복사본 추가, 실험·클래스 가중, 피처 축소, LGB 설정 변경은 모두 ±0.003 안(노이즈)이거나 악화.
  - 규칙형 피처(t1≥45/50 도달, 경과초)는 2·3단계를 개선하지 못함. 3단계 시작 조건을 찾아야 한다 → 세션 D.

### B. 이미지 — `experiments/img/` (세션 종료, 보고 완료. 자세한 표는 `experiments/img/NOTES.md`)
- **결론: 이미지는 어떤 방식으로도 센서 모델 위에 유의한 이득을 주지 못했다.**
  - CNN 단독(resnet18 미세조정): 현재 프레임 RGB 0.787 / 과거 프레임 9채널 0.748 / 의사 온도 0.777. frozen 임베딩 + LR은 0.59~0.64. 수작업 이미지 피처만 0.745.
  - 센서 모델에 CNN 확률을 스태킹: −0.001 ~ −0.017 (노이즈이거나 악화). 수작업 변화 피처 추가는 −0.013 (악화).
  - 후기 융합 `P = P_tab^(1−w) · P_cnn^w`: w=0.2에서 +0.002, w를 fold 밖에서 고르면 +0.0006 → 정직하게는 중립. 이미지를 바꾸면 예측의 약 1%가 달라진다.
  - CNN은 센서 모델과 같은 곳에서 틀린다 (정답 3 / 센서 예측 2인 966행에서 CNN 정답률 15%).
- **이유**: 팔레트가 프레임마다 자동 조정되어 색이 절대 온도가 아니다. 1→2, 2→3 경계는 표면 온도 임계값(45℃, 50℃ 이상)으로 정해져 직전·직후 프레임이 사실상 같다. 3→4에서는 프레임이 급변하는 실험이 많고, 일부 급격한 실험에서는 화염이 `meas_t1`보다 한 프레임 먼저 보인다.
- **통합 권고안**: 후기 융합 w=0.2 고정 (센서 모델과 CNN을 따로 학습). 이미지 의존을 키우려면 w=0.3이 상한.
  - 모델: timm `resnet18.tv_in1k` (torchvision ImageNet-1k, BSD-3-Clause, timm 1.0.31), 입력은 현재 프레임 RGB 128×128.
  - 전체 학습: `python experiments/img/finetune.py --mode single --tag single --full` (GPU 21초) → `~/jaehyeok/cache/img/cnn_single_full.pt` (fp16, 22.4MB). **GPU 학습은 비트 재현이 안 되므로 이 파일을 보존.**
  - 추론: `experiments/img/infer_bench.py`의 `load_model`, `cnn_probs(..., bs=1)`. **TF32 off + 1장씩 추론해야** 배치 구성에 무관하게 비트 동일 (Test 3,451장 27~39초). 기본 설정에서는 배치에 따라 확률이 4e-4까지 달라진다.
  - `requirements.txt`에 `timm` 추가 필요. `train.py`로 옮길 때는 npy 캐시 대신 PNG를 직접 읽게 바꿔야 한다.
- **아직 안 한 것 (사용자와 합의: 세션 D 결과를 본 뒤 결정)**: 4~6단계만 겨냥한 CNN, 이미지와 센서를 한 모델에 넣는 멀티모달.

### C. 순서 보정·제출 검증 — `experiments/seq/`, `experiments/checks/`
- `checks/self_check.py`: **운영진 6개 점검 자체 검사 완성.** 베이스라인 `test.py`는 6개 모두 통과. 이미지 교체 시 예측 변경 비율 15.6% (같은 실험 안에서만 교체하면 11.2%).
  - `python experiments/checks/self_check.py [--test_py ...] [--final <csv>] [--extra_args "..."] [--checks 1,2,5]`
- `seq/make_proba.py`: 기준 피처로 OOF 확률 생성 완료 → `~/jaehyeok/cache/seq/proba_inner.pkl`.
- `seq/decode.py`: `fit_params(label, time_s, exp_id)`, `decode(proba, time_s, exp_id, params, method='hmm', ...)`. 비교 스크립트 `seq/eval_decoders.py`, 로그 `seq/logs/eval_decoders.log`.
- **디코더 비교 결과 (기준 피처 LightGBM 확률, 5-fold × 3회)**: 순서 보정의 효과는 작다.
  - argmax 0.8877 (기준) / **HMM 전방 필터(alpha=1) 0.8925, +0.005, 개선 확률 0.95** / 단순 누적 최댓값 0.8831 (−0.005) / 30초 지속 조건 0.820 (크게 악화) / HMM alpha 0.5 이하는 악화.
  - 학습 fold 내부에서 설정을 고르는 방식(nested)은 argmax와 차이 없음(±0.000) → 설정을 고르지 말고 `hmm, alpha=1, clamp=True`로 고정해서 쓴다.
  - 누적 피처가 이미 행 모델에 들어가 있어서 후처리로 더 얻을 게 적은 것으로 보인다.
- **새 모델(`tab_all_nocum_r3`) 확률로 재평가: 순서 보정 효과 없음.** argmax 0.9040 / 누적 최댓값 0.9052 (+0.001, 개선 확률 0.59) / HMM 0.9038 (−0.000). → 최종 모델에 순서 보정은 넣지 않는 쪽으로 (긴 실험 점수만 0.816 → 0.840으로 오르지만 전체는 노이즈).

## 최종 제출까지 남은 일

1. A·B·C 남은 일 마무리 → 가장 좋은 조합 확정 (테이블 모델 + CNN 확률 스태킹 + 인과적 순서 보정).
2. `model.py / train.py / test.py` 3파일로 통합. CNN 가중치 포함 `weights/` 100MB 이하 목표. `test.py`는 학습·fit 없이 결정적으로.
3. `self_check.py`로 6개 점검 통과 확인. 드라이브의 공식 `check_submission.py`는 서버에 없다 → 팀원에게 받아서 한 번 더.
4. 최종 학습은 Train+Validation 97개 실험 전체로 (README에 명시).
5. README의 `<>` 칸 채우기, 사전학습 모델 출처·라이선스 기재.

## 제출 기록 (11/14: 4/30회 사용)

| 파일 (`submissions/`, 레포에 안 올라감) | 내용 | 생성 방법 | CV | Public |
|---|---|---|---|---|
| `sub01_baseline_v7.csv` | 베이스라인 v7 | `test.py` + `weights/model.joblib` | 0.810 | 0.8517 |
| `sub02_ref_gbdt.csv` | 기준 GBDT, 97개 실험 전체 학습 | `experiments/common/predict_ref_gbdt.py` | 0.886 | **0.8322** |
| `sub03_gbdt_sensor5_only.csv` | 같은 GBDT, 센서 5개 + 30초 윈도우만 (이미지·추가 컬럼 없음, 진단용) | `predict_ref_gbdt.py --groups ""` | 약 0.88 | **0.8249** |
| `sub04`~`sub07` | LR 97개 재학습 / LR+GBDT 5:5 / 5:5+HMM / LightGBM 기준+HMM (진단용) | `experiments/common/blend_lr_gbdt.py --submit` | 0.81 / 0.880 / 0.860 / 0.890 | 미제출 |
| `sub08_tab_all_nocum.csv` | 세션 A 추천 모델 | `experiments/tab/fit_predict.py` | 0.904 | **0.8309** |
| `sub09_tab_s5_nocum_rule.csv` | 세션 A 대체안 (센서 5개 + 이미지 색 통계) | `fit_predict.py --feat sensor5 --cfg '{"cum":false,"rule":true}'` | 0.900 | 미제출 |

- `sub01`~`sub03`은 보정·진단용이다. 최종 제출용이 아니다(최종은 반드시 `test.py`로 생성).
- **Public 역전 (11/14 17:20 KST)**: CV에서 0.076 높은 GBDT가 Public에서는 LR보다 0.019 낮게 나왔다. 조사 결과:
  - Test와 학습 데이터의 컬럼 분포 차이는 크지 않다 (KS 최대 0.23, `indu_t`).
  - Test에서 두 모델이 갈리는 패턴(LR 2 / GBDT 3 등)은 CV와 같고, CV에서는 그 패턴들에서 GBDT가 60~99% 맞다.
  - Test 예측의 단계별 지속 시간은 두 모델 모두 같은 배터리 구성의 학습 범위 안에 대체로 들어온다.
  - CV 예측으로 모의하면, Public이 **행 단위 무작위 30%**일 때 이런 역전 확률은 0.1%, **실험 단위 30%(3~4개 실험)**일 때 5.6%.
  - → Public은 실험 3~4개로 이뤄졌을 가능성이 높다. 그 경우 한 모델 점수의 표준편차가 0.08~0.09, 두 모델 차이의 표준편차가 0.06이라 Public만으로 모델을 고르면 위험하다. 다만 5.6%도 낮은 확률이라 실제 문제 가능성을 배제하지 않고 진단용 제출로 더 확인한다.
- **리더보드 (11/14 17:30 KST, 14팀)**: 1위 0.9125, 2위 0.9003, 3위 0.8828, 4~5위 0.8574, 6위 0.8520, 7~10위 0.8517(베이스라인 그대로, 우리 포함), 이하 0.850 / 0.838 / 0.837 / 0.673.
- **단계 지속 시간은 배터리 구성(bat_class × bat_cc × tr_method)에 따라 크게 다르다.** 예: bat 2 / cc 73은 4·5단계가 약 10초(1~2행), bat 3 / cc 28은 4단계가 360초. 4~6단계 행이 몇 개뿐인 실험이 많아 Macro F1이 몇 행에 크게 흔들린다.

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
