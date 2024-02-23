from typing import Dict, List
from util import get_char_ngrams, load_data, normalize, argmax


class NBLangIDModel:
    def __init__(self, ngram_size: int = 2, extension: bool = False):
        """
        NBLangIDModel constructor

        Args:
            ngram_size (int, optional): size of char n-grams. Defaults to 2.
            extension (bool, optional): set to True to use extension code. Defaults to False.
        """
        self._priors = None
        self._likelihoods = None
        self.ngram_size = ngram_size
        self.extension = extension

    def fit(self, train_sentences: List[str], train_labels: List[str]):
        """
        Train the Naive Bayes model (by setting self._priors and self._likelihoods)

        Args:
            train_sentences (List[str]): sentences from the training data
            train_labels (List[str]): labels from the training data
        """
       
        # COUNT N_GRAMS
        # for each element in train_sentences: call get_char_ngrams with self.n_gram_size
            # add each result to dictionaries (figure this out more)
            # add each result to our set of all 

        n_gram_counts = {}

        for i in range(len(train_sentences)):
            n_grams = get_char_ngrams(train_sentences[i], self.ngram_size)
            lang = train_labels[i]

            for n_gram in n_grams:
                
                if n_gram not in n_gram_counts:
                    n_gram_counts[n_gram] = {}
                if lang not in n_gram_counts[n_gram]:
                    n_gram_counts[n_gram][lang] = 0
                
                n_gram_counts[n_gram][lang] += 1
       
        #print(n_gram_counts)
        
        n_gram_probs = {}
        
        #loop though and +1 for LaPlace smoothing, then normalize
        for n_gram_key in n_gram_counts.keys():
            
            n_gram_values = n_gram_counts.get(n_gram_key)
            
            for lang in n_gram_values:
                
                n_gram_values[lang] += 1
            
            n_gram_probs[n_gram_key] = normalize(n_gram_values, log_prob=True)
            
        print(n_gram_counts)
        print("\n\n\n")
        print(n_gram_probs)


    def predict(self, test_sentences: List[str]) -> List[str]:
        """
        Predict labels for a list of sentences

        Args:
            test_sentences (List[str]): the sentence to predict the language of

        Returns:
            List[str]: the predicted languages (in the same order)
        """
        raise NotImplementedError

    def predict_one_log_proba(self, test_sentence: str) -> Dict[str, float]:
        """
        Computes the log probability of a single sentence being associated with each language

        Args:
            test_sentence (str): the sentence to predict the language of

        Returns:
            Dict[str, float]: mapping of language --> probability
        """
        assert not (self._priors is None or self._likelihoods is None), \
            "Cannot predict without a model!"
        raise NotImplementedError