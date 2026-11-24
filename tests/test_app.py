
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_cr002_persistent_disclaimer():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        "not verified fit classifications or hiring recommendations"
        in item.value
        for item in app.info
    )


def test_cr002_matching_method_labels():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        "V1 keyword matching" in item.value
        and "V2 TF-IDF cosine similarity" in item.value
        for item in app.caption
    )


def test_cr002_keyword_matching_limitation():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        "exact-word matching" in item.value
        and "does not recognize synonyms" in item.value
        and "does not prove that a candidate lacks a skill" in item.value
        for item in app.caption
    )
