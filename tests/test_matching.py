
import pytest
from src.matching import Matcher, keyword_score, tokens, MAX_TEXT_CHARS

@pytest.fixture
def model():
    return Matcher(['python sql analyst', 'java developer', 'nursing patient care'],
                   ['python analyst', 'java developer'])

def test_t1_all_recognized_terms():
    assert keyword_score('python sql analyst', 'python sql analyst', {'python','sql','analyst'}) == 1.0

def test_t2_no_recognized_terms():
    assert keyword_score('python', 'unknownword', {'python'}) is None

def test_t3_whole_word_java_javascript():
    assert 'java' not in tokens('javascript')
    assert keyword_score('javascript', 'java', {'java'}) == 0.0

def test_t4_identical_known_text(model):
    assert model.score('java developer', 'java developer')['v2'] == pytest.approx(1.0)

def test_t5_training_only_vocabulary(model):
    assert 'secret_test_token' not in model.vectorizer.vocabulary_
    model.score('secret_test_token', 'java developer')
    assert 'secret_test_token' not in model.vectorizer.vocabulary_

def test_t6_same_function_repeatability(model):
    assert model.score('python analyst', 'python sql analyst') == model.score('python analyst', 'python sql analyst')

def test_t7_empty_input(model):
    with pytest.raises(ValueError, match='nonempty'):
        model.score('', 'java developer')

def test_unknown_text_vector_zero(model):
    assert model.score('zzzxxyyqq', 'zzzxxyyqq')['v2'] == 0.0

def test_cr001_mismatched_input_lengths(model):
    with pytest.raises(ValueError, match="same number of entries"):
        model.score_pairs(
            ['python analyst', 'java developer'],
            ['python sql analyst']
        )


# CR-002: Input validation and boundary tests

@pytest.mark.parametrize("invalid_text", [None, 123, [], "   "])
def test_cr002_invalid_input(model, invalid_text):
    with pytest.raises(ValueError, match="Resume must be nonempty text"):
        model.score(invalid_text, "java developer")

    with pytest.raises(ValueError, match="Job description must be nonempty text"):
        model.score("python analyst", invalid_text)


@pytest.mark.parametrize("invalid_text", ["!!!", "...", "中文", "العربية"])
def test_cr002_tokenless_input(model, invalid_text):
    with pytest.raises(ValueError, match="no recognizable English letters or digits"):
        model.score(invalid_text, "java developer")

    with pytest.raises(ValueError, match="no recognizable English letters or digits"):
        model.score("python analyst", invalid_text)


def test_cr002_maximum_length_accepted(model):
    text = ("python " * MAX_TEXT_CHARS)[:MAX_TEXT_CHARS]

    assert len(text) == MAX_TEXT_CHARS
    assert model.score(text, "python analyst")["v2"] > 0
    assert model.score("python analyst", text)["v2"] > 0


def test_cr002_oversized_input_rejected(model):
    text = "x" * (MAX_TEXT_CHARS + 1)

    with pytest.raises(ValueError, match="must not exceed") as error:
        model.score(text, "java developer")

    assert text not in str(error.value)

    with pytest.raises(ValueError, match="must not exceed") as error:
        model.score("python analyst", text)

    assert text not in str(error.value)


def test_cr002_unknown_words_remain_unscoreable(model):
    result = model.score("python analyst", "unknownword")
    assert result["v1"] is None
    assert result["v2"] == 0.0
