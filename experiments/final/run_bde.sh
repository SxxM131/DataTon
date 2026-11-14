# 검증 D/E/GPU-CPU/검은 이미지 일괄 실행
set -x
cd /home/ubuntu/jaehyeok/DataTon
source ~/venv/bin/activate
D=~/data/battery_data_light
C=~/jaehyeok/cache/final
# E: train.py 2회차 (평가 생략) → 다른 폴더
python -u train.py --data_dir $D --skip_eval --weights_dir $C/weights_run2
# D: test.py 2회 (GPU)
/usr/bin/time -f "wall %e s" python test.py --input $D/test/test.csv --image_root $D/test --output $C/run1.csv --save_proba $C/run1_proba.pkl
/usr/bin/time -f "wall %e s" python test.py --input $D/test/test.csv --image_root $D/test --output $C/run2.csv --save_proba $C/run2_proba.pkl
# E: 2회차 가중치로 Test 예측
python test.py --input $D/test/test.csv --image_root $D/test --output $C/run_w2.csv --save_proba $C/run_w2_proba.pkl --weights_dir $C/weights_run2
# CPU
/usr/bin/time -f "wall %e s" python test.py --input $D/test/test.csv --image_root $D/test --output $C/run_cpu.csv --save_proba $C/run_cpu_proba.pkl --device cpu
# 검은 이미지
python test.py --input $D/test/test.csv --image_root $C/black_test --output $C/run_black.csv --save_proba $C/run_black_proba.pkl
echo ALLDONE
