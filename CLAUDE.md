# CLAUDE.md — 제6회 Uni-DTHON 데이터톤

**과제**: 열화상 이미지 + 센서 시계열로 각 관측 시점의 배터리 열폭주 **단계(1~6) 분류**
(미래 열폭주 발생·남은 시간 예측이 아님. 현재 시점 상태 분류)

- 평가지표: **Macro F1** (6개 클래스 동일 가중 → 소수 클래스도 중요)
- 리더보드: Public = Test 약 30%, Private = 약 70%. 최종 정량평가는 Private
- 수상: 일반상 = Private 80% + 심사위원 20%, 특별상(NHN) = 심사위원 100%
- **최종 모델은 열화상 이미지와 센서를 모두 사용해야 함** → 센서가 같아도 이미지가 다르면 예측이 달라져야 함 (OT). 베이스라인 그대로/하이퍼파라미터만 바꾸면 심사 점수 낮음
- **코드·규정 위반 시 수상 대상 제외** (OT)
- 심사위원 평가: 데이터 분석, 모델 설계 근거, 실험·결과 해석, 발표

## 생성형 AI 사용 (Claude 포함)

- 운영진이 OT(11/14)에서 **AI를 편하게 써도 된다**고 안내 → Claude가 데이터·로그를 직접 보면서 작업해도 됨
  (DAKER 규정 문구에는 "대회 데이터를 외부 생성형 AI에 입력 금지"가 남아 있으나, 운영진 구두 안내 기준으로 진행)
- 여전히 금지: 외부 추론 API로 예측·라벨 얻기, Test로 학습·pseudo labeling, 대회 데이터 외부 공개·재배포(GitHub 포함)

## 작업 방식

- Claude Code는 로컬(Windows, 이 폴더, 데이터 없음) 또는 서버(`~/<내이름>/DataTon`)에서 실행
- 학습·추론은 서버에서만 (로컬에서 학습·추론 금지). 로컬에서 고쳤으면 `git push` → 서버 `git pull`
- 서버는 팀 공유 계정 → 11/15 13:50 서버 반납 전에 Claude Code `/logout` 또는 `~/.claude` 삭제
- **모든 커밋·push는 `jaehyeok` 브랜치에.** `main`에 직접 push 금지. 최종 제출 전(11/15 13:30 전)에 `main`으로 merge

## 데이터

`~/data/battery_data_light/` (서버) — train, validation, test, sample_submission.csv. 이미지는 CSV의 `image_path`로 연결

| 컬럼 | 설명 |
|---|---|
| `sample_id` | 샘플 고유 ID (제출 키) |
| `image_path` | 열화상 이미지 경로 |
| `meas_v` | 전압 (V) |
| `meas_a` | 전류 (A) |
| `meas_t1` | 표면 온도 (℃) |
| `meas_t2` | 양극부 온도 (℃) |
| `meas_p` | 압력 |
| `experiment_id` | 실험 익명 ID |
| `relative_time_s` | 실험 첫 관측 기준 경과 시간 (초) |
| `label` | 1~6 (train/validation만) |

데이터 규칙:
- 각 행 = 한 관측 시점. 관측 간격이 불규칙 → 시계열은 `experiment_id` + `relative_time_s` 기준으로 구성
- 윈도우는 **행 수가 아니라 초(relative_time_s) 기준** 권장
- 사용 가능: 같은 실험 내 **현재 + 과거** 관측(센서·이미지), `relative_time_s`
- 사용 금지: 미래 관측, 실험 전체 기준 값(평균·최댓값·길이·순위), 다른 실험 관측, 라벨 및 라벨 파생값
- 결측 대체·정규화 등 전처리 통계는 **train에서만** 산출 → val/test엔 적용만
- 자체 검증은 **experiment_id 단위 분할** 권장
- Train + Validation은 학습에 사용 가능 (단, 성능 검증은 학습에 안 쓴 데이터로. 발표에 구성 명시)
- Test는 추론에만. Pseudo labeling, 외부 데이터, 외부 추론 API 금지
- 사전학습 모델: 대회 전 공개된 범용 가중치(torchvision, timm, HF ImageNet 등) 허용 → README에 출처·버전·라이선스
- 언어는 Python만. **학습·추론은 제공 GPU 서버에서만** (OT)

## 제출 파일 형식

- `sample_id,label` 열 순서 그대로, 행 순서 = `sample_submission.csv`, 인덱스 저장 금지 (`index=False`)
- label은 **1~6 정수**. 내부 인덱스 0~5나 확률을 그대로 내지 말 것
- 누락·중복·추가 행·결측 없음
- 리더보드 제출: 11/14 15:00 ~ 11/15 13:00, **팀당 1일 30회** (자정 초기화)
- 최종 파일 1개 선택 (미선택 시 Public 최고점). Private 동점이면 최종 파일을 먼저 낸 팀이 위

## 마감 (11/15 일)

| 시각 | 할 일 |
|---|---|
| 11/14 (1일차) | GitHub 레포 생성, 코드 수시 push, GitHub PAT 미리 발급 |
| 11/15 11:00 | `model.py / train.py / test.py` 3파일 구조 정리 완료 |
| 11/15 13:00 전 | `check_submission.py` 전부 ✓ 확인 |
| **11/15 13:00** | **리더보드 마감** — 최종 파일 1개 선택 |
| **11/15 13:30** | **코드·PPT·GitHub 레포 메일 제출 마감** (OT 기준. DAKER·노션의 13:50은 오기). 이후 커밋 반영 안 됨 |
| 11/15 13:50 | 서버 회수 |
| 11/15 14:00 | 발표 시작 |
| 11/15 17:00 | 최종 심사·결과 발표 |

메일 (`unidsw@gmail.com`, 레포 링크 + PPT를 한 통에):
```
제목: [Uni-D 데이터톤] 팀명 최종 제출
팀명:
GitHub 레포:
리더보드에서 선택한 최종 파일명:
첨부: 발표 PPT (25MB가 넘으면 드라이브 링크)
```

## 제출 레포 구조 (필수)

```
model.py          # 피처·모델 정의 (resolve_image_paths, extract_image_features, build_features, build_model)
train.py          # 학습 → weights/ 에 모델 + 학습한 전처리(스케일러·PCA 등) 저장
test.py           # weights/ 로드 → 예측 → CSV 저장. 학습·fit 절대 없음
weights/          # 100MB 초과는 다운로드 링크를 README에 (웹 업로드는 파일당 25MB 제한)
requirements.txt  # torch는 버전 고정 금지 (torch>=2.7 처럼 하한만)
README.md         # 실행 방법, 환경(OS·Python·GPU), 학습·검증 구성, 시계열 피처 구성, 사전학습 모델(출처·버전·라이선스)
```

- 베이스라인(`uni_d_baseline_v7`)의 README 양식과 3파일 템플릿을 기준으로 작성
- **현재 코드 = 베이스라인 v7 그대로** (가중치 경로만 `weights/`로 변경): 센서 5개 + 이미지 그레이스케일 통계 6개(mean·std·max·p95·p99·hot_ratio) + 과거 30초 윈도우 mean·max·delta → Imputer + StandardScaler + LogisticRegression. class_weight none/balanced 중 Validation Macro F1 높은 쪽 선택. 이대로 내면 심사 점수 낮음 → 개선 대상
- 운영진은 `test.py`만 실행, `train.py`는 읽기만. 라이브러리 로딩 포함 오류 없이 돌아야 함
- 파일 더 나눠도 됨. 앙상블은 가중치 전부 넣고 test.py에서 합치기
- 대회 데이터는 레포에 올리지 않음 (`.gitignore`에 `data/`). 레포는 Public (결과 발표 후 private 가능)

## test.py 재현성 규칙 — 코드 작성 시 항상 지킬 것

```bash
python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>
```
그 밖의 인자는 전부 기본값을 둔다. 재학습 없이 Private Score를 복원할 수 있어야 함.

운영진 자가 점검(`check_submission.py`) 항목:

| 점검 | 요구사항 | 흔한 원인 |
|---|---|---|
| 최종 파일 재현 | test.py 출력 == DAKER에 낸 파일 | 다른 가중치, 노트북 캐시 |
| 같은 입력 재실행 | 매번 동일 결과 | 시드 미고정, `model.eval()` 누락 |
| 행 순서 섞기 | 입력 순서 무관 | 정렬 없이 행 번호로 시계열 계산 |
| 실험 나눠 넣기 | 다른 실험 유무와 무관 | Test 전체 통계, Test에서 재fit, GPU 배치 크기 영향 |
| 미래 행 빼기 | 실험별 앞 60%만 넣어도 각 행 결과 동일 | 실험 전체 max·길이, 뒤쪽 관측 사용 |
| 이미지 바꾸기 (OT) | 센서가 같아도 이미지가 다르면 예측이 달라야 함 | 이미지를 무시하거나 이미지 피처 영향이 사실상 0 |

따라서:
- 입력 CSV에 있는 행만 보고 예측. 시계열 피처는 실험별로 `relative_time_s` 정렬 후 **현재까지의 과거 값만** (causal)
- 출력은 입력 CSV의 원래 행 순서로 되돌려 저장
- 스케일러/PCA 등은 train에서 fit → weights/ 저장 → test에서 load만
- 노트북에서 미리 만든 피처 파일 읽기 금지
- 시드 고정, PyTorch는 `model.eval()` + `torch.no_grad()`, 결과가 배치 구성·크기에 의존하지 않게
- 리더보드 최종 파일은 **반드시 test.py로 생성**. 제출할 때마다 그 결과를 만든 가중치 보존

저장 예:
- sklearn 계열: `joblib.dump({'model': model, 'columns': list(X.columns)}, 'weights/model.joblib')`
- PyTorch: `torch.save(model.state_dict(), 'weights/model.pt')` → test.py에서 `load_state_dict` 후 `model.eval()`

자가 점검 (드라이브 폴더의 `check_submission.py`를 레포에 두고 서버에서):
```bash
python check_submission.py --data_dir ~/data/battery_data_light --final <DAKER에 낼 최종 CSV>
```

## GPU 서버

- 팀당 RTX 5080 (16GB) 1대, Ubuntu, 계정 `ubuntu`, 포트 22, **팀원 전원이 같은 계정 공유**
- 접속 정보(IP·비밀번호·와이파이·드라이브)는 팀 대표가 문자로 받음. **이 파일·레포·외부에 절대 쓰지 말 것**
- 대회 와이파이에 노트북만 연결 (휴대폰 X). Windows에 ssh 없으면: 설정 → 앱 → 선택적 기능 → OpenSSH 클라이언트

| 내용 | 경로 |
|---|---|
| Python venv (torch, JupyterLab) — 팀 공용 | `~/venv` → `source ~/venv/bin/activate` |
| 대회 데이터 | `~/data/battery_data_light` |
| 베이스라인 원본 (수정 금지, 복사해서 사용) | `~/data/baseline/uni_d_baseline_v7` |
| 각자 작업 폴더 | `~/<내이름>` |

초기 세팅 (팀에서 한 명만): 드라이브의 `setup.sh` 맨 위 `DATA_URL`, `BASELINE_URL`에 각 **파일** 공유 링크를 넣고 `bash setup.sh` → 마지막에 `GPU: True` 나오면 완료.
- `$'\r': command not found` → `sed -i 's/\r$//' setup.sh`

레포 실행 (서버, `~/<내이름>/DataTon`에 clone되어 있음):
```bash
source ~/venv/bin/activate
cd ~/<내이름>/DataTon && git checkout jaehyeok && git pull
pip install -r requirements.txt
D=~/data/battery_data_light
python -u train.py --data_dir $D 2>&1 | tee train.log   # tmux 안에서 → weights/model.joblib
python test.py --input $D/test/test.csv --image_root $D/test --output submission.csv
```

서버 규칙:
- 긴 학습은 **tmux 안에서** (`tmux new -s train-<이름>`, 빠져나오기 Ctrl+B → D, 복귀 `tmux attach -t <이름>`)
- 학습 전 `nvidia-smi`로 다른 팀원 사용 확인 (GPU 1장 공유, OOM 주의). 안 쓰는 노트북 커널은 종료
- `sudo apt upgrade` 금지 (GPU 드라이버 깨짐). 필요한 것만 `apt install`
- `rm -rf` 주의 — 본인 폴더 밖 건드리지 않기
- GitHub 토큰을 서버에 저장하지 말 것. `git config`는 `--global` 없이 레포 안에서만
- 서버가 교체될 수 있음 → 코드는 수시로 GitHub에 push, 가중치·로그는 13:30 전에 노트북/드라이브로 백업
