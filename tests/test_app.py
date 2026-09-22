"""대시보드 스모크 테스트: python -m pytest tests/test_app.py  (또는 python tests/test_app.py)"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_all(verbose: bool = False) -> None:
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    if verbose:
        print("metrics:", [(m.label, m.value) for m in at.metric])
        print(at.dataframe[0].value.to_string()); print(at.dataframe[1].value.to_string())

    for opt in at.sidebar.radio[0].options[1:]:          # 과거 충격 프리셋 전부
        at.sidebar.radio[0].set_value(opt).run()
        assert not at.exception, (opt, at.exception)
        if verbose:
            print(opt, "->", [m.value for m in at.metric][4:7])

    side = next(x for x in at.radio if x.label == "포지션")
    for v in ("import", "export"):
        side.set_value(v).run()
        assert not at.exception, (v, at.exception)
        if verbose:
            print(v, at.success[0].value); print(at.dataframe[2].value.to_string())


def test_app():
    run_all()


if __name__ == "__main__":
    run_all(verbose=True)
