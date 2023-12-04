import pytest
from src.preproc import preprocess_comments


def test_label_length(predictor, vectorizer, dim_reducer):
    """Predictor returns label with 6 classes"""
    label_a = get_label(text="Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!",
                        predictor=predictor,
                        vectorizer=vectorizer,
                        dim_reducer=dim_reducer)
    assert len(label_a) == 6


def test_vectorizer_numerical_1024(vectorizer):
    """Vectorizer converts text to array of numbers of length 1024"""
    vect = vectorizer.transform(["Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"])
    vect, = vect.toarray().tolist()
    assert len(vect) == 1024 and all(isinstance(element, (int, float)) for element in vect)


def test_reducer_reduces(vectorizer, dim_reducer):
    """Dimensionality reducer reduces the dimentionality of the vector"""
    vect = vectorizer.transform(["Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"])
    vect = vect.toarray()
    vect_reduced = dim_reducer.transform(vect)
    assert vect.shape[0] >= len(vect_reduced)


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
                                   ("GET FUCKED UP. GET FUCKEEED UP. GOT A DRINK THAT YOU CANT PUT DOWN??? GET FUCK UP GET FUCKED UP.", [0, 0, 1, 0, 0, 0])])
def test_class_correctly_classified(inp_a, label, predictor, vectorizer, dim_reducer):
    """Model predicts classes correctly"""
    label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer, dim_reducer=dim_reducer)
    assert label_a[label.index(1)] == 1


def get_label(text, predictor, vectorizer, dim_reducer):
    text_preproc, = preprocess_comments([text])
    vect = vectorizer.transform([text_preproc])
    vect_reduced = dim_reducer.transform(vect.toarray())
    res, = predictor.predict(vect_reduced)
    return res.tolist()
