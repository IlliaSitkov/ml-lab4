from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import nltk
import zipfile
import os

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
