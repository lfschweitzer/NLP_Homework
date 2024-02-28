# Homework 2 Report

## Basics
External resources used on this assignment:
* Class worksheet
* Various google searches for python syntax

Classmates you talked to about this assignment:
* Just each other

How many hours did you spend on this assignment?
* 11

### Group Member #1
* Your middlebury email: iethier@middlebury.edu
* How many hours did you spend on outside of class on CS 457 this week, _excluding this assignment_? 5

### Group Member #2
_Delete this section if you worked alone_
* Your middlebury email: lschweitzer@middlebury.edu
* How many hours did you spend on outside of class on CS 457 this week, _excluding this assignment_? 4.5-5.5

## Report
### Evaluating evaluation metrics
#### What is one advantage of using accuracy over a confusion matrix to evaluate your model?
Accuracy is an easier method to quickly assess our model. Accuracy is straightforward and easy to understand, whereas a confusion matrix can be complex and harder to understand.


#### What is one advantage of using a confusion matrix over accuracy to evaluate your model?
A confusion matrix gives us a more thorough way to assess the performance of our model. It allows us to see specifically what is happening, occasionally reveal information where accuracy as an assessment measurement might be misleading. For example, if a training data set is unbalanced and 90% of the sentences are in english, a model that always predicts english will have a 90% accuracy rate on our training set. However, a confusion matrix would show that the model is consistently choosing english and therefore reveal that our model is making invalid predictions.


#### Interpreting your confusion matrix
_Choose a language from the data set. Given any knowledge that you have about properties about that language (e.g., writing system, language families), explain any unique properties of its row in the confusion matrix, including (a) performance compared to other languages, (b) languages that it is more likely to be predicted as than others, and (c) anything else you think is relevant._

In our confusion matrix there is a lot of overlap between Spanish and Italian. More specifically the model often misidentified spanish as italian. This could be for a variety of reasons. They both have Latin roots and because of that they share a lot of vocabulary and grammatical rules. 


### Effects of priors
Update your code so that your prior is uniform, e.g., the probability for each of the 8 languages is $log\left(\frac{1}{8}\right)$. Then, answer the following questions.
#### How do your results change? Why?

We added self.prior = {lang: math.log(1 / 8) for lang in lang_counts.keys()} to line 60 so that our prior would be uniform for all 8 languages. Before making this change, we got an accuracy score of 0.9912527365395538 on the full data set. After setting self.prior to be uniform, we had an accuracy score of 0.9907846124609047. We believe this nominal difference is due to the self.prior value being a very small part of the many values being multiplied by each other (or added for log calculations). Our formula gives self.prior the same weight as each ngram in the sentence so changing the value won't make a huge difference.

#### Describe a situation in which you might _want_ to use a uniform prior.

While calculating self.prior does not take a lot of effort compared to the rest of the code, if a developer did want to have slightly fewer calculations in their code it is an easy step to remove. There also may be situations were a developer wants to continually add sentences to their training data without recalculating how many they have of each language -- allowing them to more dynamically expand their machine's capabilities without having to redo every calculation. Another scenario could be that you want to avoid introducing bias from your training set into your model. If you have no reason to favor one label over another you would want the prior to be uniform even if there are more of one type than the other in the training set. Additionally, if you have incomplete information, uniform prior could be a reasonable assumption.