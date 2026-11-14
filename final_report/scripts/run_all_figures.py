"""figures/ 전체 생성 (재학습 없음)."""
import runpy
from pathlib import Path

here = Path(__file__).resolve().parent
for name in [
    "01_lb_history.py",
    "02_pipeline.py",
    "03_timeseries_stage.py",
    "04_transition_align.py",
    "05_confusion.py",
    "06_rejected_delta.py",
    "07_pseudotemp.py",
]:
    print("===", name)
    runpy.run_path(str(here / name), run_name="__main__")
print("ALL DONE")
