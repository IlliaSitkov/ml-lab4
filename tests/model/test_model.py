import pytest
from src.main import preprocess_comments


@pytest.mark.parametrize("inp_a, inp_b, label", [
    ("Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!",
     "Stupid peace of shit stop deleting my stuff asshole go die", [1, 0, 1, 0, 1, 0])])
def test_invariance(inp_a, inp_b, label, predictor, vectorizer, dim_reducer):
    """Shorter sentence with the same meaning should be classified as the original sentence"""
    label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer, dim_reducer=dim_reducer)
    label_b = get_label(text=inp_b, predictor=predictor, vectorizer=vectorizer, dim_reducer=dim_reducer)
    assert label_a == label_b == label


@pytest.mark.parametrize("inp_a, label", [("I will return in a few hours Canuck to wreak havoc on your faggot canadian ass", [0, 0, 0, 0, 1, 0]),
                                   ("Hi! I am back again! Last warning! Stop undoing my edits or die!", [1, 0, 0, 0, 0, 0]),
                                   ("What a motherfucking piece of crap those fuckheads for blocking us!", [0, 1, 0, 0, 0, 0]),
                                   ("GET FUCKED UP. GET FUCKEEED UP. GOT A DRINK THAT YOU CANT PUT DOWN??? GET FUCK UP GET FUCKED UP.", [0, 0, 1, 0, 0, 0]),
                                   ("Fuck you, Smith. Please have me notified when you die. I want to dance on your grave.", [0, 0, 0, 1, 0, 0]),
                                   ("I am going to murder ZimZalaBim ST47 for being evil homosexual jews.", [0, 0, 0, 0, 0, 1])])
def test_class_correctly_classified(inp_a, label, predictor, vectorizer, dim_reducer):
    """Model predicts classes correctly"""
    label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer, dim_reducer=dim_reducer)
    assert label_a[label.index(1)] == 1
#
#
# @pytest.mark.parametrize("input, label", [("Hello world", 1), ("Hello earth", 1)])
# def test_mft(input, label, predictor, vectorizer):
#     """Minimum Functionality Tests (simple input/output pairs)"""
#     prediction = get_label(text=input, predictor=predictor, vectorizer=vectorizer)
#     assert label == prediction
#
#


def get_label(text, predictor, vectorizer, dim_reducer):
    text_preproc, = preprocess_comments([text])
    vect = vectorizer.transform([text_preproc])
    vect_reduced = dim_reducer.transform(vect.toarray())
    res, = predictor.predict(vect_reduced)
    return res.tolist()
