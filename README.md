# Uni-D 배터리 열폭주 6단계 분류 — 17팀

열화상 이미지와 센서 시계열로 각 관측 시점의 열폭주 단계(1~6)를 분류합니다.

| 순서 | 구성 요소 | 하는 일 |
|---|---|---|
| ① | 테이블 모델 | 센서와 이미지 색 통계의 시계열 피처 266개 + 규칙 피처 30개 → LightGBM 6단계 확률 |
| ② | 이미지 모델 | 현재·직전 열화상 프레임과 그 차이 → CNN(resnet18) 6단계 확률 |
| ③ | 융합 | 두 확률을 0.8 : 0.2로 결합 (`P ∝ P_tab^0.8 · P_cnn^0.2`) |
| ④ | 2·3단계 재판정 | 2 또는 3으로 나온 행만, 압력 상승 피처 48개를 더한 두 번째 LightGBM으로 다시 판정 |
| ⑤ | 규칙 보정 | 5·6단계 예측은 그 단계의 이벤트를 관측한 뒤에만 인정하고, 4·5단계는 되돌아가지 않게 함 |

## 팀 정보
- 팀명: 17팀
- 최종 선택 제출 파일: `sub18_final_v9.csv`, Private 복원 대상 (`python test.py`의 출력과 바이트 단위로 같음)

## 개발 환경
- OS: Ubuntu 22.04.5 LTS (jammy)
- Python: 3.10.12
- GPU: 대회 제공 GPU 서버, NVIDIA GeForce RTX 5080 (16GB), 드라이버 580.178.04, CUDA 12.8
- 라이브러리: `requirements.txt` 참고 (실행 확인 버전: torch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.31, lightgbm 4.7.0,
  scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, pillow 12.3.0, joblib 1.6.0)

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 피처 생성(이미지 색 통계, 시계열·규칙·압력 상승 피처), LightGBM 설정, CNN 구조·전처리·추론, 융합, 2·3단계 재판정, 규칙 보정 |
| `train.py` | Train으로 학습해 Validation으로 평가, Train+Validation 전체로 최종 학습, 가중치 저장 |
| `test.py` | `weights/` 로드 → 입력 CSV에 들어 있는 행만으로 추론 → 제출 파일 생성 (학습·fit 없음) |
| `weights/tab_lgbm.joblib` | LightGBM 기본 모델 + 2·3단계 재판정 모델 + 피처 열 순서 + 배터리 구성별 압력 범위 중앙값 표 + 설정 |
| `weights/cnn_pairdiff.pt` | 이미지 CNN 가중치 |
| `requirements.txt` | 필요한 라이브러리 |

`experiments/`는 실험 코드와 로그이고 최종 추론에 쓰이지 않습니다. 대회 데이터와 제출 CSV는 레포에 올리지 않았습니다.

## 실행 방법
```bash
pip install -r requirements.txt

# 추론 (가중치는 기본값 weights/ 사용, GPU가 있으면 GPU, 없으면 CPU)
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --output submission.csv
# (선택) --sample_submission <경로>/sample_submission.csv : 제출 양식 행 순서로 저장 (기본은 입력 CSV 행 순서)

# 학습 (선택)
python train.py --data_dir <battery_data_light 경로>              # Validation 평가 + Train+Validation 최종 학습
python train.py --data_dir <battery_data_light 경로> --eval_cnn   # + Train만으로 CNN을 학습해 융합 점수까지 평가 (GPU)
python train.py --data_dir <battery_data_light 경로> --train_cnn  # 최종 CNN도 새로 학습 (GPU)
```
- `test.py`는 학습·fit을 하지 않고 입력 CSV에 있는 행만 보고 피처를 만듭니다. 행 순서를 섞거나, 실험 일부만 넣거나,
  실험별 앞부분만 넣어도 같은 행에는 같은 예측이 나옵니다. Test 3,451행 추론은 GPU에서 약 11초, CPU에서 약 30초입니다.
- 추론은 결정적 설정입니다 (`model.eval()` + `torch.no_grad()`, `cudnn.deterministic=True`, 이미지 1장씩 추론).
  GPU와 CPU의 예측 라벨은 Test 전체에서 같았습니다.
- 운영진 `check_submission.py`의 모든 항목을 통과했습니다. 모든 이미지를 검은 이미지로 바꾸면 예측의 1.30%가 바뀝니다.
- LightGBM 학습은 결정적 설정이라 `train.py`를 다시 실행하면 같은 테이블 모델이 나옵니다.
- CNN의 GPU 학습은 비트 단위로 재현되지 않습니다. 제출 결과를 만든 가중치는 레포의 `weights/cnn_pairdiff.pt`이고
  (`experiments/img/finetune2.py --full`로 학습), `train.py`의 기본 실행은 CNN 가중치를 다시 만들지 않습니다.
  `--train_cnn`을 주면 같은 설정으로 새로 학습하고, 그 가중치는 조금 다를 수 있습니다.

## 학습·검증 구성
- 학습 데이터: Train + Validation 97개 실험 전체 (최종 학습).
- 검증 방법
  - 교차검증: Train+Validation 97개 실험을 실험 단위로 층화(bat_class × tr_method)해 5-fold × 3회 반복.
    Test는 500행이 넘는 실험을 500행으로 줄여 두었으므로, 검증 행도 같은 방식으로 줄이고 그 행들만 보고 피처를 만들어 채점했습니다.
  - 제공 분할: Train 78개 실험으로 학습 → Validation 19개 실험 평가 (`train.py --eval_cnn` 출력).
- 검증 점수 (Macro F1)
  - 교차검증: 0.9229 (이미지 융합까지 넣으면 0.9234). 단계별 F1은 1.000 / 0.832 / 0.912 / 0.932 / 0.947 / 0.914.
  - 제공 Validation: 0.9272 (Test와 같은 방식으로 줄인 6,514행), 0.9178 (원본 10,200행).
- 2·3단계 재판정과 규칙 보정의 구조와 상수는 Validation을 포함한 97개 실험을 보고 정했습니다.
  따라서 위 점수는 모델에 대해서만 hold-out이고 규칙에 대해서는 아닙니다.

## 시계열 피처 구성
모든 시계열 피처는 같은 `experiment_id` 안에서 `relative_time_s`로 정렬한 뒤 현재 시점 이하의 관측만 사용합니다. 윈도우는 행 수가 아니라 초 단위입니다.
- 과거 윈도우 통계 (110개): 센서 4개(`meas_v`, `meas_t1`, `meas_t2`, `meas_p`), 이미지 색 통계 3개, `thermal_t`, `meas_co`, `meas_co2`, `indu_t`의
  과거 30/120/300초 구간 평균·최댓값·최솟값과 120초 표준편차.
- 변화량·변화율 (110개): 같은 신호의 15/30/60/120/300초 전(그 시각 이하 마지막 관측) 대비 변화량과 초당 변화율.
- 이벤트 경과 시간 (12개): 온도가 정해진 값에 처음 도달한 뒤, 전압이 지금까지 최댓값의 일정 비율 아래로 처음 내려간 뒤 경과 초.
- 규칙 피처 (30개): 온도 도달, 전압 급락, 압력 상승, 온도 합(`meas_t1 + meas_t2`)의 지금까지 최댓값과 그 이후 하강 등 고정 임계값 기반 피처.
  직전 관측 대비 차분과 지금까지의 누적값만 사용합니다.
- 압력 상승 피처 (48개, 2·3단계 재판정 모델 전용): 5/10/15/20/30초 전 대비 압력 상승량과 그 누적 최댓값, 임계값을 처음 넘은 뒤 경과 초.
  상승량은 학습 데이터에서 구한 배터리 구성별 압력 범위 중앙값, 또는 그 실험의 첫 관측 압력으로 나눕니다.
  학습 데이터에서 산출해 `weights/`에 저장하는 통계는 이 중앙값 표 하나입니다.
- 시계열이 아닌 입력: 원시 값 23개(센서, 가스, 온도, 배터리 메타)와 프레임마다 독립으로 계산한 이미지 색 통계 11개.
- 이미지 CNN: 현재 프레임과 같은 실험의 직전 관측 프레임, 두 프레임의 차이만 사용합니다.
- 규칙 보정: 같은 실험의 현재와 과거 관측, 고정 상수만 사용합니다.
- 미래 시점 관측, 실험 전체 기준 통계(평균·최댓값·길이·순위), 다른 실험의 관측은 사용하지 않습니다.

## 사용한 사전학습 모델
- `resnet18.tv_in1k`: timm 1.0.31에서 불러온 torchvision ImageNet-1k 사전학습 가중치 (원 출처 torchvision `ResNet18_Weights.IMAGENET1K_V1`)
  - 출처: https://huggingface.co/timm/resnet18.tv_in1k (timm), torchvision
  - 라이선스: BSD-3-Clause
  - 첫 합성곱은 timm `in_chans=9` 옵션으로 9채널 입력에 맞게 변환하고, 분류 헤드는 새로 학습했습니다.
- 추론(`test.py`)은 구조만 만들고(`pretrained=False`) `weights/cnn_pairdiff.pt`를 올리므로 네트워크 접속이 필요 없습니다.
