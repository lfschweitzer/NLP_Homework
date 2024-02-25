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
        lang_counts = {}
        languages = []

        for i in range(len(train_sentences)):
            n_grams = get_char_ngrams(train_sentences[i], self.ngram_size)
            lang = train_labels[i]
            
            if lang not in languages: #running list of all languages
                languages.append(lang)
            
            if lang not in lang_counts:
                lang_counts[lang] = 1
            else:
                lang_counts[lang] += 1

            for n_gram in n_grams:
                
                if n_gram not in n_gram_counts:
                    n_gram_counts[n_gram] = {}
                if lang not in n_gram_counts[n_gram]:
                    n_gram_counts[n_gram][lang] = 0
                
                n_gram_counts[n_gram][lang] += 1

        print("n gram counts: ", n_gram_counts, "\n")

        n_gram_probs = {}
        
        #loop though and +1 for LaPlace smoothing, then normalize
        for n_gram_key in n_gram_counts.keys():
            
            for lang in languages:
                
                if lang not in n_gram_counts[n_gram_key]:
                    n_gram_counts[n_gram_key][lang] = 1
                else:
                    n_gram_counts[n_gram_key][lang] += 1
            
            print("n gram counts after adding one: ", n_gram_counts[n_gram_key], "\n")
            
            n_gram_probs[n_gram_key] = normalize(n_gram_counts[n_gram_key], log_prob=False)
        
        print("n_gram_count post adding 1 and normalizing", n_gram_probs, "\n")
        
        # self._likelihoods is a dict of dict where [to: [spanish: # of to's in spanish+1/# of ngrams], etc.]
        self._likelihoods = n_gram_probs
        
        for lang in lang_counts.keys():
            lang_counts[lang] /= len(train_sentences)
        
        #self.prior is a dict of span: # of sentences in spanish / total # of sentences
        self.prior = lang_counts
        
        print("self likelihoods: ", self._likelihoods, "\n self.prior:", self.prior, "\n")

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
            
            lang_likelihood = self.predict_one_log_proba(test_sentence) #should normally be test sentences
            
            for lang_key in lang_likelihood.keys():
                
                lang_likelihood[lang_key] *= self.prior[lang_key]          

            predictions.append(argmax(lang_likelihood))
            
        print("predictions", predictions)
                                     
                    

    def predict_one_log_proba(self, test_sentence: str) -> Dict[str, float]:
        """
        Computes the log probability of a single sentence being associated with each language

        Args:
            test_sentence (str): the sentence to predict the language of

        Returns:
            Dict[str, float]: mapping of language --> probability
        """
        # assert not (self._priors is None or self._likelihoods is None) #"Cannot predict without a model!"
        
        lang_likelihood = {} # for each lang the likelihood of the sentence
            
        n_grams = get_char_ngrams(test_sentence, self.ngram_size) # ngrams in sentence
        print("nGrams", n_grams)
        
        for n_gram in n_grams:
            
            if n_gram in self._likelihoods:
                
                for lang in self._likelihoods[n_gram]: #for each language that the ngram could be
                    
                    if lang not in lang_likelihood:
                        
                        lang_likelihood[lang] = self._likelihoods[n_gram][lang]
                    else:
                        lang_likelihood[lang] += self._likelihoods[n_gram][lang] #add log probs
        
        print("likelhood of each sentence:", lang_likelihood, "\n") 
                       
        return lang_likelihood
