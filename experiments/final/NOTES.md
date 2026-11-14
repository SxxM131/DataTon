# 세션 E — 최종 파이프라인 통합 기록

목표: 루트 `model.py / train.py / test.py / weights/ / requirements.txt / README.md`를 최종 모델(테이블 LightGBM 266+30 피처 + pairdiff CNN 후기 융합 w=0.2)로 교체.

## 진행 로그
- 09:00 UTC 시작. 실험 코드 확인: exp01.color_stats, tab/feats.build, rules/rules.rule_features, rules/run_rules.py(LGB 설정), img/finetune2.py, img/infer_pair.py.
  - LGB 파라미터(rules_tab_nocum_r3): run_rules.LGB_DEFAULT(n_jobs=4) + fast(lr .1, 200, colsample .3, max_bin 63), random_state 42.
  - 융합식(img/latefuse.py): log(Pt+1e-9)*(1-w) + log(Pc+1e-6)*w → argmax.
  - 데이터 확인: (실험, 시각) 중복 0, relative_time_s 전부 정수 → 정렬 동률 없음.
- 루트 model.py 작성 (실험 코드 복사, experiments import 없음). 정렬 동률은 sample_id로 고정하도록만 바꿈 (데이터에 동률 없음 → 값 동일).
- **검증 A 통과** (`experiments/final/verify_a.py`, 로그 `verify_a.log`): pool/evalp/test 피처 296개 값 완전 일치(NaN 포함),
  색 통계 300장 캐시와 완전 일치, 공통 CV rep0 0.9029 = rules_tab_nocum_r3 rep0 0.9029, 예측 100% 일치, 확률 차이 0.
  (rules_tab_nocum_r3 의 0.9074는 3회 반복 평균, rep0 단독은 0.9029)
- test.py, train.py 작성. weights: tab_lgbm.joblib (LGB + 열 순서 + 설정), cnn_pairdiff.pt (기존 cnn_pairdiff_full.pt 복사).
  train.py --eval_cnn 실행 중 (Train만으로 CNN 학습해 융합 점수 계산).
- train.py --eval_cnn 1회차 (92초, 로그 `train_run1.log`): 제공 분할 Train→Validation
  - 원본 행(10,200): 테이블 0.8431 / CNN 단독 0.7055 / 융합 0.8345
  - Test식 다운샘플(6,514): 테이블 0.8834 / CNN 단독 0.7581 / 융합 0.8758
  - 융합이 이 분할에서는 −0.008~−0.009 (Train만으로 새로 학습한 CNN 1회, 2단계 F1이 떨어짐). CV 기준 융합 효과는 +0.0001(중립)로 보고됨.
  - weights/tab_lgbm.joblib, weights/cnn_pairdiff.pt 생성. 기존 weights/model.joblib 삭제.
- README 작성(점수·환경·피처·사전학습 모델 채움, 최종 파일명만 자리표시). requirements.txt에 lightgbm/torch>=2.7/torchvision/timm 추가.
- `run_bde.sh` 실행 중: train.py 2회차(평가 생략, 30초) → test.py GPU 2회(각 약 11초) → 2회차 가중치 → CPU → 검은 이미지. 비교는 `compare_runs.py`.
- **검증 D/E/CPU/검은 이미지** (`compare_runs.log`):
  - D: test.py GPU 2회 → 파일 바이트 동일, tab/cnn/융합 확률 비트 동일. 실행 시간 약 11.4~11.6초(벽시계, 로딩 포함), CPU 강제 29.8초.
  - E: train.py 2회(별도 폴더 weights_run2) → Test 테이블 확률 비트 동일.
  - CPU vs GPU: 라벨 차이 0/3451, CNN 확률 최대차 1.2e-6, 융합 최대차 1.6e-7.
  - 검은 이미지(같은 크기, ~/jaehyeok/cache/final/black_test): 예측 변경 181/3451 = 5.2% (색 통계도 바뀌어 테이블 확률도 변함).
  - 융합이 테이블 단독과 다른 행 0.58% (Test). sub08 대비 다른 행 60/3451 = 1.74% (테이블 단독 vs sub08: 1.56%, 규칙 피처 추가 효과).
  - submissions/sub10_final_v1.csv = run1.csv. self_check 실행 중.
- CNN 이식 확인 (`verify_cnn_port.py`): model.cnn_probs == infer_pair.cnn_probs 비트 동일 (Test 전체, GPU).
- self_check (--final sub10): (1)(2)(3) PASS 확인, 나머지 진행 중 (로그 `self_check.log`).
- self_check --final sub10 (halves): (1)~(6) 전부 PASS. (6) 이미지 무작위 교체 시 예측 변경 1.9%, 같은 실험 안에서만 교체 2.1%. 실험 하나씩(each) 점검 진행 중.
- self_check 실험 하나씩(each, 11회): PASS (로그 `self_check.log`). → 검증 A~E 모두 완료.

## 남은 위험·메모
- 제공 분할(Train→Validation)에서는 융합이 테이블 단독보다 −0.008 (Train만으로 새로 학습한 CNN 1회). CV에서는 +0.0001(중립). 이미지 사용 요건용.
- 규칙 상수는 97개 실험 전체에서 정해져 CV가 약간 낙관적일 수 있음 (세션 D 메모).
- 정렬 동률(같은 실험·같은 시각)은 sample_id로 고정 (실험 코드는 입력 순서 의존이었음, 실제 데이터에는 동률 없음).
- `_lag_index`/`_lag`의 key = 실험코드×1e7 + t → relative_time_s가 1e7초를 넘으면 실험 간 섞임 (현재 최대 7,710초, 문제 없음).
- train.py --eval_cnn / --train_cnn 은 timm 사전학습 가중치(HF Hub, 로컬 캐시 있음)가 필요. test.py는 네트워크 불필요.
