import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from joblib import dump


def train_model():
    train_dataset = pd.read_csv("/kaggle/input/kma-ml2/kmaml223/train.csv")

    test_dataset = pd.read_csv("/kaggle/input/kma-ml2/kmaml223/test.csv")
    test_comments = test_dataset["comment_text"].values

    train_dataset = train_dataset.sample(n=100000, random_state=42)
    train_comments = train_dataset["comment_text"].values
    train_labels = train_dataset.iloc[:, 2:].to_numpy()

    train_comments_preproc = preprocess_comments(train_comments)
    test_comments_preproc = preprocess_comments(test_comments)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, strip_accents="unicode", stop_words="english")

    train_feature_vectors_tfidf = tfidf_vectorizer.fit_transform(train_comments_preproc).toarray()

    test_feature_vectors_tfidf = tfidf_vectorizer \
        .transform(test_comments_preproc).toarray()

    pca = PCA(n_components=0.99)
    reduced_train_pca = pca.fit_transform(train_feature_vectors_tfidf)

    reduced_test4_pca = pca.transform(test_feature_vectors_tfidf)

    X_train, X_test, y_train, y_test = train_test_split(
        reduced_train_pca, train_labels, test_size=0.2, random_state=42
    )

    classifier = MultiOutputClassifier(LogisticRegression())

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    test_pred = classifier.predict(reduced_test4_pca)

    columns = [
        "id",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    ids = test_dataset["id"].values

    result = np.column_stack((ids, *test_pred.T)).tolist()

    df = pd.DataFrame(result, columns=columns)

    df.to_csv("sample_submission.csv", index=False)

    dump(classifier, "classifier.joblib")
    dump(tfidf_vectorizer, "vectorizer.joblib")
    dump(pca, 'pca_model.joblib')
