#!/bin/bash
# 세션 A 3차: nocum 기준 세부 제거 실험 (빠른 설정, rep1). --drop '*xxx' = 이름에 xxx 포함 피처 제외
source ~/venv/bin/activate; cd ~/jaehyeok/DataTon; export OMP_NUM_THREADS=6
FAST='{"learning_rate":0.1,"n_estimators":200,"colsample_bytree":0.3,"max_bin":63}'
R="python -u experiments/tab/run.py --n_repeats 1 --no_official --lgb $FAST"
NC='{"cum":false}'
if [ "$1" = A ]; then
$R --name f_all_cumv --feat all --drop '*_cummax,*_cummin,*_d0,*_fromcmax,*_since_cmax'
$R --name f_all_nocum_w120 --feat all --cfg '{"cum":false,"windows":[30,120],"delta_windows":[15,30,60,120]}'
$R --name f_all_nocum_nothresh --feat all --cfg '{"cum":false,"thresh":false}'
$R --name f_s5_nocum --feat sensor5 --cfg "$NC"
else
$R --name f_all_nocum_wclass --feat all --cfg "$NC" --weight class
$R --name f_all_nocum_pooltl --feat all --cfg "$NC" --train_rows pool+tl
$R --name f_all_nocum_tl --feat all --cfg "$NC" --train_rows tl
$R --name f_all_nocum_noimgts --feat all --cfg '{"cum":false}' --drop 'pt_mean_,pt_p90_,pt_frac50_'
fi
