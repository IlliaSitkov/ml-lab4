import pytest
from joblib import load


@pytest.fixture(scope="module")
def predictor(request):
    predictor_loc = "data/classifier.joblib"
    model = load(predictor_loc)
    return model


@pytest.fixture(scope="module")
def vectorizer(request):
    vectorizer_loc = "data/vectorizer.joblib"
    vectorizer = load(vectorizer_loc)
    return vectorizer


@pytest.fixture(scope="module")
def dim_reducer(request):
    dim_reducer_loc = "data/pca_model.joblib"
    dim_reducer = load(dim_reducer_loc)
    return dim_reducer
