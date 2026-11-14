#!/bin/bash
source ~/venv/bin/activate
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
python -u finetune2.py --tag pairdiff 2>&1 | grep -v Warn
python -u finetune2.py --tag pairdiff_sens --sens 1 2>&1 | grep -v Warn
python -u finetune2.py --tag sensonly --sens 1 --noimg 2>&1 | grep -v Warn
