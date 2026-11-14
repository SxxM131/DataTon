#!/bin/bash
# 스태킹 배치: $1 = base (ref|tab). 스레드 2개.
source ~/venv/bin/activate
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 NT=2
python -u stack.py --tag temporal --base $1 --variants probs
until grep -q '^done' logs/finetune_single.log; do sleep 10; done
python -u stack.py --tag single --base $1 --variants probs,agg
until grep -q '^done' logs/finetune_temporal_pt.log; do sleep 10; done
python -u stack.py --tag temporal_pt --base $1 --variants probs
