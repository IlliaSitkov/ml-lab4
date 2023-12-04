import pytest
from joblib import load


@pytest.fixture(scope="module")
def predictor(request):
    predictor_loc = "data/my_model.joblib"
    model = load(predictor_loc)
    return model


@pytest.fixture(scope="module")
def vectorizer(request):
    vectorizer_loc = "data/vectorizer.joblib"
    vectorizer = load(vectorizer_loc)
    return vectorizer
