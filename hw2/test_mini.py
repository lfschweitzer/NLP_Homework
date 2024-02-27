import math

from model import NBLangIDModel
from scoring import accuracy_score, confusion_matrix
from util import print_confusion_matrix


def main():
    """
    Use this test script to test your model on the "mini" data set, which includes
    the example from class.
    """
    # data (yes, these are single words, not actual sentences)
    train_sentences = ["ablaze", "hablo", "learn"]
    train_labels = ["eng", "spa", "eng"]
    test_sentences = ["able"]

    # train model and predict language for the one test sentence
    model = NBLangIDModel(ngram_size=2)
    model.fit(train_sentences, train_labels)
    results = model.predict_one_log_proba(test_sentences[0])

    # convert from log probabilities
    print({lang: math.e ** log_prob
           for lang, log_prob in results.items()})

    # added this test for accuracy_score
    y_true = ["spa", "eng", "spa"]
    y_pred = ["eng", "eng", "spa"]
    print(accuracy_score(y_true, y_pred))

    y_true = ["spa", "eng", "spa"]
    y_pred = ["eng", "eng", "spa"]
    print_confusion_matrix(confusion_matrix(y_true, y_pred, ["spa", "eng", "fra"]), ["spa", "eng", "fra"])

if __name__ == "__main__":
    main()
