#!/bin/bash
# 세션 A 1차 스크리닝 배치 (n_repeats=1). 큐 A/B 두 줄로 병렬 (각 n_jobs=6)
source ~/venv/bin/activate; cd ~/jaehyeok/DataTon; export OMP_NUM_THREADS=6
R="python -u experiments/tab/run.py --n_repeats 1"
SMALL='{"windows":[30],"win_stats":["mean","max"],"std_windows":[],"delta_windows":[30,120]}'
if [ "$1" = A ]; then
$R --name tab_s5_pool --feat sensor5
$R --name tab_all_small --feat all --cfg "$SMALL"
$R --name tab_s5_small --feat sensor5 --cfg "$SMALL"
$R --name tab_all_wclass --feat all --weight class
else
$R --name tab_all_pooltl --feat all --train_rows pool+tl
$R --name tab_all_wexp --feat all --weight exp
$R --name tab_all_tl --feat all --train_rows tl
$R --name tab_all_pooltl3 --feat all --train_rows pool+tl3
fi
