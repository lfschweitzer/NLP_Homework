# Homework 2 Report

## Basics
External resources used on this assignment:
* Class worksheet
* Various google searches for python syntax

Classmates you talked to about this assignment:
* Just each other

How many hours did you spend on this assignment?
* 9

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

-- lots of overlap between Spanish and Italian


### Effects of priors
Update your code so that your prior is uniform, e.g., the probability for each of the 8 languages is $log\left(\frac{1}{8}\right)$. Then, answer the following questions.
#### How do your results change? Why?



#### Describe a situation in which you might _want_ to use a uniform prior.


