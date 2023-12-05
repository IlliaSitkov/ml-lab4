import sys
import requests
from io import BytesIO
import joblib
from preproc import preprocess_comments


def load_model_from_github(model_name):
    github_url = f'https://github.com/illiasitkov/ml-lab4/raw/main/data/{model_name}.joblib'
    response = requests.get(github_url)
    if response.status_code == 200:
        model = joblib.load(BytesIO(response.content))
        return model
    else:
        print(f"Failed to fetch {model_name} from GitHub.")
        sys.exit(1)


def classify_text(text, predictor, vectorizer, dim_reducer):
    text_preproc, = preprocess_comments([text])
    vect = vectorizer.transform([text_preproc])
    vect_reduced = dim_reducer.transform(vect.toarray())
    res, = predictor.predict(vect_reduced)
    return res.tolist()


def main():
    classifier = load_model_from_github('classifier')
    vectorizer = load_model_from_github('vectorizer')
    pca_model = load_model_from_github('pca_model')

    if len(sys.argv) < 2:
        print("Please provide the text to classify as a command-line argument")
        sys.exit(1)

    text_to_classify = sys.argv[1]

    predictions = classify_text(text_to_classify, classifier, vectorizer, pca_model)

    print("Predictions:")

    for class_name, class_value in zip(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], predictions):
        print(f"{class_name}: {'False' if class_value == 0 else 'True'}")


if __name__ == "__main__":
    main()
