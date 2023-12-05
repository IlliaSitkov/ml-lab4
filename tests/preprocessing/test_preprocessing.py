from src.preproc import preprocess_comments, lemmatize_text, clean_comments
import string


def test_lemmatization():
    """Lemmatization should normalize words"""
    text = "The quick brown foxes are jumping over the lazy dogs."
    text_lem = lemmatize_text(text)
    assert len(text) > len(text_lem)


def test_text_cleaning():
    """Function should remove punctuation marks from text"""
    text = "Hello, World! This is an example sentence with lots of punctuation: commas, semicolons; colons! parentheses (and brackets), question marks? exclamation marks! and dashes - all present."
    text_clean, = clean_comments([text])
    assert all(punctuation not in text_clean for punctuation in string.punctuation)


def test_preprocessing():
    """Function should remove punctuation marks from text and perform lemmatization"""
    text = "Hello, World! This is an example sentence with lots of punctuation: commas, semicolons; colons! parentheses (and brackets), question marks? exclamation marks! and dashes - all present."
    text_preprocessed, = preprocess_comments([text])
    punctuation_count = sum([1 if char in string.punctuation else 0 for char in text])
    assert all(punctuation not in text_preprocessed for punctuation in string.punctuation) \
           and len(text) - punctuation_count > len(text_preprocessed)
