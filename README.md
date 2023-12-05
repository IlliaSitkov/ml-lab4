# ML Course Final Lab Repository

## Overview

This repository contains the code for the final lab project of the Machine Learning course. The main goal of this project was to train a classifier to categorize comments into six classes. Each comment may belong to more than one class simultaneously. The target classes are as follows:

1. Toxic
2. Severe Toxic
3. Obscene
4. Threat
5. Insult
6. Identity Hate

## Repository structure
```
root/
├── .github/workflows/           # GitHub Actions configurations
├── data/                        # Trained models
├── EDA/                         # Data analysis and statistics notebooks
├── src/                         # Source code
│   ├── classifier.py            # Inference script
│   ├── main.py                  # Script for training models
│   ├── preproc.py               # Preprocessing code
├── tests/                       # Testing scripts
└── requirements.txt             # Dependencies
```

## Trained Models

The trained models used for the classification task are stored in the `data` directory. The following models are included:

- **classifier.joblib:** A trained classifier using MultiOutputClassifier with LogisticRegression as the base estimator. It was trained on a dataset of 80,000 comments.

- **vectorizer.joblib:** A trained TfIdfVectorizer, which converts text comments into vectors with 1024 features. It was trained on a dataset of 100,000 comments.

- **pca_model.joblib:** A trained PCA (Principal Component Analysis) model, which reduces the dimensionality of the 1024-dimensional vectors to 866 dimensions. It was trained on a dataset of 100,000 comments.

## Usage

To use the trained models, you can load them into your Python environment and apply them to new data for classification. Below is an example of how to load the models using the joblib library:

```python
import joblib

# Load the models:
predictor = joblib.load('data/classifier.joblib')
vectorizer = joblib.load('data/vectorizer.joblib')
dim_reducer = joblib.load('data/pca_model.joblib')

# Usage example:
vect = vectorizer.transform(["Your comment here"])
vect_reduced = dim_reducer.transform(vect.toarray())
result, = predictor.predict(vect_reduced)
```

## Inference Script

You can use the inference script `classifier.py` located in the `src` folder to classify comments. To use it, run the following command in your terminal:

```bash
python src/classifier.py "Your comment to classify here"
```

Example response:
```
Predictions:
toxic: True
severe_toxic: True
obscene: True
threat: False
insult: True
identity_hate: False
```

## Preprocessing

Before performing inference, the input text undergoes preprocessing steps, including cleaning from punctuation and lemmatization. The corresponding code for these preprocessing tasks can be found in the `src/preproc.py` file.

## Development mode

To get started with the project, follow these initial steps:

1. **Install Python 3.9:** Make sure you have Python 3.9 installed on your system. If not, you can download it from [python.org](https://www.python.org/).

2. **Install Dependencies:** Run the following command in your terminal to install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Testing and Code Quality

The codebase is thoroughly tested, with a test coverage of 57%. You can run the tests using the following commands:

```bash
# Run all tests
pytest tests

# Run tests with coverage
pytest --cov=src
```
Additionally, a Flake8 style checker is set up to ensure code quality. You can run it using the following command:

```bash
flake8
```

## Continuous Integration with GitHub Actions

The project includes pre-configured GitHub Actions workflows that run tests and style checks automatically when changes are pushed to the main branch. This ensures continuous integration, helping maintain code quality and consistency.

The workflows can be found in the `.github/workflows/` directory.
