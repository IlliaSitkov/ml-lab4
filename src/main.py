import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from joblib import dump

nltk.download("punkt")
nltk.download("wordnet", "lib")
wordnet_zip_path = "lib/corpora/wordnet.zip"
wordnet_extract_path = "lib/corpora/"
nltk.data.path.append("lib")

if not os.path.exists(
        os.path.join(wordnet_extract_path, "corpora", "wordnet")
):
    with zipfile.ZipFile(wordnet_zip_path, "r") as zip_ref:
        zip_ref.extractall(wordnet_extract_path)


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def clean_comments(comments):
    translator = str.maketrans("", "", string.punctuation)
    comments_clean = [comment.translate(translator) for comment in comments]
    return comments_clean


def preprocess_comments(comments):
    comments_clean = clean_comments(comments)
    comments_lem = [lemmatize_text(comment) for comment in comments_clean]
    return comments_lem


def train_model():
    train_dataset = pd.read_csv("/kaggle/input/kma-ml2/kmaml223/train.csv")

    test_dataset = pd.read_csv("/kaggle/input/kma-ml2/kmaml223/test.csv")
    test_comments = test_dataset["comment_text"].values

    train_dataset = train_dataset.sample(n=100000, random_state=42)
    train_comments = train_dataset["comment_text"].values
    train_labels = train_dataset.iloc[:, 2:].to_numpy()

    train_comments_preproc = preprocess_comments(train_comments)
    test_comments_preproc = preprocess_comments(test_comments)

    tfidf_vectorizer = TfidfVectorizer(max_features=1024, strip_accents="unicode", stop_words="english")

    train_feature_vectors_tfidf = tfidf_vectorizer.fit_transform(train_comments_preproc).toarray()

    test_feature_vectors_tfidf = tfidf_vectorizer \
        .transform(test_comments_preproc).toarray()

    pca4 = PCA(n_components=0.95)
    reduced_train4_pca = pca4.fit_transform(train_feature_vectors_tfidf)

    reduced_test4_pca = pca4.transform(test_feature_vectors_tfidf)

    X_train, X_test, y_train, y_test = train_test_split(
        reduced_train4_pca, train_labels, test_size=0.2, random_state=42
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

    dump(classifier, "my_model.joblib")
    dump(tfidf_vectorizer, "vectorizer.joblib")
    dump(pca4, 'pca_model.joblib')
