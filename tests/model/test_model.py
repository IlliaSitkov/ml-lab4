import pytest


@pytest.mark.parametrize("inp_a, inp_b, label", [
    ("Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!",
     "Stupid peace of shit stop deleting my stuff asshole go die", [1, 1, 1, 0, 1, 0])])
def test_invariance(inp_a, inp_b, label, predictor, vectorizer):
    """Invariance via verb injection (changes should not affect outputs)"""
    label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer)
    label_b = get_label(text=inp_b, predictor=predictor, vectorizer=vectorizer)
    assert label_a == label_b == label

# @pytest.mark.parametrize("inp_a, inp_b, label", [("Hello world", "Hello earth", 1)])
# def test_invariance(inp_a, inp_b, label, predictor, vectorizer):
#     """Invariance via verb injection (changes should not affect outputs)"""
#     label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer)
#     label_b = get_label(text=inp_b, predictor=predictor, vectorizer=vectorizer)
#     assert label_a == label_b == label
#
#
# @pytest.mark.parametrize("input, label", [("Hello world", 1), ("Hello earth", 1)])
# def test_mft(input, label, predictor, vectorizer):
#     """Minimum Functionality Tests (simple input/output pairs)"""
#     prediction = get_label(text=input, predictor=predictor, vectorizer=vectorizer)
#     assert label == prediction
#
#


def get_label(text, predictor, vectorizer):
    vect = vectorizer.transform([text])
    return predictor.predict(vect)
