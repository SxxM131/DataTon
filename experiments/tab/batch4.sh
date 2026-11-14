#!/bin/bash
# 세션 A 4차: 추천 후보 3반복(+official) + LGB 파라미터 비교(rep1)
source ~/venv/bin/activate; cd ~/jaehyeok/DataTon; export OMP_NUM_THREADS=6
FAST='{"learning_rate":0.1,"n_estimators":200,"colsample_bytree":0.3,"max_bin":63}'
R="python -u experiments/tab/run.py"
if [ "$1" = A ]; then
$R --n_repeats 3 --lgb $FAST --name tab_all_nocum_r3 --feat all --cfg '{"cum":false}'
$R --n_repeats 3 --lgb $FAST --name tab_s5_nocum_r3 --feat sensor5 --cfg '{"cum":false}'
else
$R --n_repeats 1 --no_official --name f_all_nocum_slow --feat all --cfg '{"cum":false}' --lgb '{"max_bin":63}'
$R --n_repeats 1 --no_official --name f_all_nocum_reg --feat all --cfg '{"cum":false}' --lgb '{"learning_rate":0.05,"n_estimators":300,"num_leaves":15,"min_child_samples":100,"colsample_bytree":0.3,"max_bin":63,"reg_lambda":5.0}'
$R --n_repeats 1 --no_official --name f_all_nocum_seed3 --feat all --cfg '{"cum":false}' --lgb $FAST --seeds 3
fi
