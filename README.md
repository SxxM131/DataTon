# Uni-D 배터리 열폭주 6단계 분류 — 17팀

열화상 이미지 + 센서 시계열로 각 관측 시점의 열폭주 단계(1~6)를 분류합니다.
센서 시계열 LightGBM(테이블 피처 266개 + 규칙 피처 30개)과 열화상 CNN(현재·직전 프레임 비교)을 후기 융합합니다.

## 팀 정보
- 팀명: 17팀
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: Ubuntu 22.04.5 LTS (jammy)
- Python: 3.10.12
- GPU: 대회 제공 GPU 서버, NVIDIA GeForce RTX 5080 (16GB), 드라이버 580.178.04, CUDA 12.8
- 주요 라이브러리 (실행 확인 버전): torch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.31, lightgbm 4.7.0,
  scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, pillow 12.3.0, joblib 1.6.0 (`requirements.txt` 참고)

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 이미지 경로 연결, 이미지 색 통계, 테이블·규칙 피처, LightGBM 설정, CNN 구조·전처리·결정적 추론, 융합 |
| `train.py` | Train → Validation 평가, Train+Validation 전체 최종 학습, 가중치 저장 |
| `test.py` | `weights/` 로드 → 입력 CSV의 행만으로 추론 → 제출 파일 생성 (학습·fit 없음) |
| `weights/tab_lgbm.joblib` | LightGBM 모델 + 피처 열 순서 + 피처·모델·융합 설정 |
| `weights/cnn_pairdiff.pt` | 이미지 CNN 가중치 (fp16 state_dict, 추론 시 float32로 올림) |

## 실행 방법
```bash
pip install -r requirements.txt

# 추론 (가중치는 기본값 weights/ 사용, GPU가 있으면 GPU, 없으면 CPU)
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --output submission.csv
# (선택) --sample_submission <경로>/sample_submission.csv : 제출 양식 행 순서로 저장 (기본은 입력 CSV 행 순서)

# 학습 (선택) → 평가 점수 출력 + weights/tab_lgbm.joblib 저장 + CNN 가중치 준비
python train.py --data_dir <battery_data_light 경로>              # 테이블 평가 + 최종 학습
python train.py --data_dir <battery_data_light 경로> --eval_cnn   # + Train만으로 CNN 학습해 융합 점수까지 (GPU)
python train.py --data_dir <battery_data_light 경로> --train_cnn  # 최종 CNN도 새로 학습 (GPU)
```
- `test.py`는 학습·fit을 하지 않습니다. 입력 CSV에 있는 행만 보고 피처를 만들기 때문에 행 순서를 섞거나,
  실험 일부만 넣거나, 실험별 앞부분만 넣어도 같은 행에는 같은 예측이 나옵니다. 출력은 입력 CSV의 행 순서입니다.
- 실행 시간 (Test 3,451행, 라이브러리 로딩 포함): GPU 약 11초, CPU만 사용 시 약 30초. GPU와 CPU의 예측 라벨은 Test 전체에서 같았습니다
  (CNN 확률 차이 최대 1.2e-6).
- CNN 추론은 결정적 설정입니다: `model.eval()` + `torch.no_grad()`, TF32 끔, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, **이미지 1장씩 추론**(배치 구성에 따른 수치 차이 제거).
- LightGBM 학습은 결정적 설정(`deterministic=True`, `force_row_wise=True`, `random_state=42`, `n_jobs=4`)이라
  `train.py`를 다시 실행해도 같은 테이블 모델이 나옵니다.
- **CNN 가중치에 대해**: GPU 학습은 비트 단위로 재현되지 않습니다. 그래서 `train.py`는 기본적으로 제출에 사용한
  학습 완료 가중치를 `weights/cnn_pairdiff.pt`로 복사하고, CNN 재학습은 `--train_cnn`을 줄 때만 합니다
  (같은 코드·설정이지만 결과 가중치는 조금 다를 수 있음). **제출 결과를 만든 가중치는 레포의 `weights/cnn_pairdiff.pt`입니다.**

## 모델 구성
1. **테이블 모델**: LightGBM 다중 분류 (learning_rate 0.1, 200 trees, num_leaves 31, min_child_samples 40,
   subsample 0.8, colsample_bytree 0.3, max_bin 63, reg_lambda 1.0). 입력은 아래 시계열 피처 266개 + 규칙 피처 30개.
2. **이미지 모델 (pairdiff CNN)**: timm `resnet18` (ImageNet 사전학습) 미세조정. 입력 9채널 =
   현재 프레임 RGB + 같은 실험의 직전 관측 프레임 RGB + 두 프레임의 차이, 128×128. 출력 6단계 확률.
   - 학습: Train+Validation 전체, 직전 프레임은 5/10/15초 전 중 무작위 (Test의 10·15초 간격 대응),
     좌우 반전·크기·이동 증강, 클래스 가중 1/n, 긴 실험은 500행 상한 비중, 8 epoch(1,496 step, 배치 128), AdamW + OneCycle.
   - 추론: 직전 프레임 = 입력 CSV에 있는 같은 실험의 행 중 시간상 바로 앞 관측 (실험 첫 행은 자기 자신).
3. **융합**: `P ∝ P_tab^0.8 · P_cnn^0.2` (가중치 0.2 고정) → `label = argmax + 1`. 순서 보정·클래스 가중은 쓰지 않습니다
   (검증에서 이득 없음).

## 학습·검증 구성
- **최종 학습 데이터: Train + Validation 97개 실험 전체** (원본 5초 간격 행 그대로, 증강·샘플 가중 없음).
- **검증 방법**
  - 주 검증: Train+Validation 97개 실험을 실험 단위로 층화(bat_class × tr_method) 5-fold × 3회 반복.
    Test는 500행이 넘는 실험을 500행으로 균등 다운샘플해 두었으므로, 검증 행도 같은 방식으로 다운샘플하고
    검증 행의 시계열 피처는 그 행들만 보고 만듭니다 (Test에서 보이는 조건과 동일).
  - 보조 검증: 대회 제공 분할 (Train 78개 실험으로 학습 → Validation 19개 실험 평가).
- **검증 점수 (Macro F1)**
  - 5-fold × 3회 교차검증 (테이블 모델, Test식 다운샘플 행 채점): **0.9074** (실험 단위 부트스트랩 90% 구간 0.887~0.924).
    이미지 CNN 융합(w=0.2)은 같은 교차검증에서 +0.0001로 점수에는 중립이었습니다 — 이미지 사용 요건을 위해 넣었고,
    가중치 0.2는 검증 결과를 보고 그 이상 키우지 않은 값입니다.
  - 제공 분할 Train 78개 → Validation 19개 실험 (`train.py --eval_cnn` 출력):

    | 검증 행 | 테이블 단독 | CNN 단독 | 융합 (최종 방식) |
    |---|---|---|---|
    | Test식 다운샘플 (6,514행) | 0.8834 | 0.7581 | 0.8758 |
    | Validation 원본 행 (10,200행) | 0.8431 | 0.7055 | 0.8345 |

    (제공 Validation은 19개 실험뿐이라 점수 흔들림이 큽니다. 원본 행 점수는 긴 실험 몇 개가 행의 66%를 차지해 Test와 가중이 다릅니다.)

## 시계열 피처 구성
모든 시계열 피처는 같은 `experiment_id` 안에서 `relative_time_s`로 정렬한 뒤 **현재 시점 이하의 관측만** 사용합니다(인과적).
윈도우는 행 수가 아니라 **초 단위**입니다 (학습 데이터의 5초 간격과 Test의 10·15초 간격에서 같은 의미).
미래 관측, 실험 전체 기준 통계(평균·최댓값·길이·순위), 다른 실험의 관측, 학습된 정규화 통계는 쓰지 않습니다.
- **원시 값 (23)**: `relative_time_s`, 센서 4개(`meas_v`, `meas_t1`, `meas_t2`, `meas_p`; `meas_a`는 항상 0이라 제외),
  가스 10종, `atmp_t`, `thermal_t`, `indu_t`(+결측 여부), 배터리 메타 4개. `indu_t.1`은 Test에서 전부 결측이라 쓰지 않습니다.
- **이미지 색 통계 (11)**: 열화상의 무지개 의사색상을 색조(hue)로 의사 온도(0~1)로 되돌린 뒤 평균·표준편차·분위수·최댓값·
  임계 초과 비율·흰색 비율. 이미지마다 독립 계산, 고정 상수만 사용.
- **과거 윈도우 통계 (110)**: 센서 4개 + 이미지 신호 3개(`pt_mean`, `pt_p90`, `pt_frac50`) + `thermal_t`, `meas_co`, `meas_co2`, `indu_t`
  에 대해 과거 30/120/300초 구간 `(t−W, t]`의 평균·최댓값·최솟값, 120초 표준편차.
- **변화량·변화율 (110)**: 같은 신호의 15/30/60/120/300초 전(그 시각 이하 마지막 관측) 대비 변화량과 초당 변화율.
- **이벤트 경과 시간 (12)**: `meas_t1`이 45/60/80/100/150/300℃, `meas_t2`가 100/200℃에 처음 도달한 뒤 경과 초,
  전압이 지금까지 최댓값의 90/50/10% 아래로 처음 내려간 뒤, 0.05V 아래로 내려간 뒤 경과 초.
- **규칙 피처 (30, `r_*`)**: 학습 데이터에서 라벨 전환 시점을 분석해 찾은 상수 임계값 기반 피처.
  `meas_t1` 45/50℃ 도달 후 경과 초, 압력 상승, 직전 관측 대비 전압 급락(−0.56V 이하 또는 2.5V 미만) 이후 경과 초와 횟수,
  배터리 종류별 5단계 조건(전압 1.355V 이하 / 0V / 온도 상승률), `t1+t2`의 지금까지 최댓값과 그 이후 하강,
  조건이 발화한 가장 높은 단계(`r_stage`) 등. 직전 관측 기준 차분과 초당 변화율, 지금까지의 누적값만 사용합니다.
- **이미지 CNN의 시간 정보**: 현재 프레임과 같은 실험의 직전 관측 프레임(과거)의 차이만 사용합니다.

## 사용한 사전학습 모델
- `resnet18.tv_in1k` — timm 1.0.31에서 불러온 torchvision ImageNet-1k 사전학습 가중치 (원 출처 torchvision `ResNet18_Weights.IMAGENET1K_V1`)
  - 출처: https://huggingface.co/timm/resnet18.tv_in1k (timm), torchvision
  - 라이선스: BSD-3-Clause
  - 첫 합성곱은 timm `in_chans=9` 옵션으로 9채널 입력에 맞게 변환, 분류 헤드는 새로 학습
- 추론(`test.py`)은 구조만 만들고(`pretrained=False`) `weights/cnn_pairdiff.pt`를 올리므로 네트워크 접속이 필요 없습니다.
