#!/bin/bash
# 세션 A 2차 스크리닝 (빠른 LGB 설정, rep1, official 생략). 클래스 2/3 하락 원인 피처 그룹 제거 실험
source ~/venv/bin/activate; cd ~/jaehyeok/DataTon; export OMP_NUM_THREADS=6
FAST='{"learning_rate":0.1,"n_estimators":200,"colsample_bytree":0.3,"max_bin":63}'
R="python -u experiments/tab/run.py --n_repeats 1 --no_official --lgb $FAST"
SMALL='{"windows":[30],"win_stats":["mean","max"],"std_windows":[],"delta_windows":[30,120]}'
if [ "$1" = A ]; then
$R --name f_all --feat all
$R --name f_all_nocum --feat all --cfg '{"cum":false}'
$R --name f_all_wexp --feat all --weight exp
else
$R --name f_all_small --feat all --cfg "$SMALL"
$R --name f_all_nothresh --feat all --cfg '{"thresh":false}'
$R --name f_s5_small --feat sensor5 --cfg "$SMALL"
fi
