from typing import Dict, List
from util import get_char_ngrams, load_data, normalize, argmax
import math
from collections import defaultdict, Counter


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

        ngram_counts = defaultdict(lambda: defaultdict(int)) # dict of dict
        lang_counts = Counter(train_labels) # count of sentences of each lang for use in self.prior
        ngram_list = set() # set of all ngrams

        for i in range(len(train_sentences)):
            ngrams = get_char_ngrams(train_sentences[i], self.ngram_size)
            lang = train_labels[i]
            
            for ngram in ngrams:
                
                ngram_list.add(ngram) #running list of all ngrams
                
                ngram_counts[lang][ngram] += 1

        ngram_probs = {}
        
        #loop though and +1 for LaPlace smoothing, then normalize
        for lang_key in ngram_counts.keys():
            
            for unique_ngram in ngram_list:
        
                ngram_counts[lang_key][unique_ngram] += 1
            
            #normalize probabilities
            ngram_probs[lang_key] = normalize(ngram_counts[lang_key], log_prob=True)
        
        self._likelihoods = ngram_probs
   
        #self.prior is a dict of span: # of sentences in spanish / total # of sentences
        self.prior = {lang: math.log(lang_count / len(train_sentences)) for lang, lang_count in lang_counts.items()}

    def predict(self, test_sentences: List[str]) -> List[str]:
        """
        Predict labels for a list of sentences

        Args:
            test_sentences (List[str]): the sentence to predict the language of

        Returns:
            List[str]: the predicted languages (in the same order)
        """
        predictions = []
        
        for test_sentence in test_sentences:
            
            lang_likelihood = self.predict_one_log_proba(test_sentence)
                        
            predictions.append(argmax(lang_likelihood))

        return predictions                          
                    

    def predict_one_log_proba(self, test_sentence: str) -> Dict[str, float]:
        """
        Computes the log probability of a single sentence being associated with each language

        Args:
            test_sentence (str): the sentence to predict the language of

        Returns:
            Dict[str, float]: mapping of language --> probability
        """
        
        lang_likelihood = {} # for each lang the likelihood of the sentence
            
        ngrams = get_char_ngrams(test_sentence, self.ngram_size) # ngrams in sentence
        
        for ngram in ngrams:
            
            for lang in self._likelihoods:
                
                if lang not in lang_likelihood:
                    
                    if ngram in self._likelihoods[lang]:
                        lang_likelihood[lang] = self._likelihoods[lang][ngram]
                else:
                    
                    if ngram in self._likelihoods[lang]: # if we dont have ngram in training, ignore
                        
                        lang_likelihood[lang] += self._likelihoods[lang][ngram] # addition cause log probs
       
               
        for lang_key in lang_likelihood.keys():
                
            lang_likelihood[lang_key] += self.prior[lang_key] # addition cause log probs
            
        # if no evidence for any language because all ngrams are unseen
        if not lang_likelihood:
            # return probabilities that reflect only the priors
            return {lang: self.prior[lang] for lang in self.prior}
                            
        return lang_likelihood       
        
