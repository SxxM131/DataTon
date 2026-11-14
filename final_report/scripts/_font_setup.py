"""공통 스타일. 서버에 한글 폰트 없어 그림 라벨은 영문 사용."""
import matplotlib as mpl


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })
