# Deep Learning with Python

| Title            | Deep Learning with Python |
|------------------|---------------------------|
| Author           | François Chollet          |
| Publication Date | 2017                      |
| Page Count       | 384                       |
| Genre            | Non-Fiction               |
| ISBN             | 978-1617294433            |
| Date Read        | 2023-04-18                |
| Date Finished    | NA                        |


## Summary

### Part 1: Fundamentals of deep learning

### Chapter 1: What is deep learning?

#### 1.1 Artificial intelligence, machine learning, and deep learning

#### 1.1.1 Artificial Intelligence 

- The field of Artificial Intelligence can be described as the effort to automate intellectual tasks normally performed by humans

- AI is a general field that encompasses machine learning and deep learning, but that also includes many more approaches that don’t involve any learning like hard coding

- Although symbolic AI proved suitable to solve well-defined, logical problems, such as playing chess, it turned out to be intractable to figure out explicit rules for solving more complex, fuzzy problems, such as image classification, speech recognition, and language translation

#### 1.1.2 Machine learning

- In classical programming, the paradigm of symbolic AI, humans input rules (a program) and data to be processed according to these rules, and out come answers. 

- With machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce original answers.

- A machine-learning system is trained rather than explicitly programmed.  

#### 1.1.3 Learning representations from data

- To achieve machine learning, we need three things:

1. Input data points
2. Examples of the expected output—In a speech-recognition task, these could be human-generated transcripts of sound files. In an image task, expected outputs could be tags such as “dog,” “cat,” and so on.
3. A way to measure whether the algorithm is doing a good job representations are different ways to look at data Learning, in the context of machine learning, describes an automatic search  process for better representations.
 
- Machine-learning algorithms aren’t usually creative in finding these transformations; they’re merely searching through a predefined set of  operations, called a hypothesis space.

#### 1.1.4 The “deep” in deep learning

- The deep in deep learning isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations

- In deep learning, these layered representations are (almost always) learned via models called neural networks

- You can think of a deep network as a multistage information-distillation operation, where information goes through successive filters and comes out increasingly purified

#### 1.1.5 Understanding how deep learning works, in three figures

- The specification of what a layer does to its input data is stored in the layer’s weights, which in essence are a bunch of numbers

- In technical terms, we’d say that the transformation implemented by a layer is parameterized by its weights

- In this context, learning means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets.

- The loss function takes the predictions of the network and the true target(what you wanted the network to output) and computes a distance score, capturing how well the network has done on this specific example

- The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example. This adjustment is the job of the optimizer, which implements what’s called the Backpropagation algorithm: the central algorithm in deep learning

- Initially, the weights of the network are assigned random values, so the network merely implements a series of random transformations. Naturally, its output is far from what it should ideally be, and the loss score is accordingly very high. But with every example the network processes, the weights are adjusted a little in the correct direction, and the loss score decreases. 

- This is the training loop, which, repeated a sufficient number of times (typically tens of iterations over thousands of examples), yields weight values that minimize the loss function

#### 1.1.6 What deep learning has achieved so far

 In particular, deep learning has achieved the following breakthroughs, all in historically difficult areas of machine learning:
1. Near-human-level image classification
2. Near-human-level speech recognition
3. Near-human-level handwriting transcription
4. Improved machine translation
5. Improved text-to-speech conversion
6. Digital assistants such as Google Now and Amazon Alexa
7. Near-human-level autonomous driving
8. Improved ad targeting, as used by Google, Baidu, and Bing
9. Improved search results on the web
10. Ability to answer natural-language questions
11. Superhuman Go playing

#### 1.1.7 Don’t believe the short-term hype

Although deep learning has led to remarkable achievements in recent years, expectations for what the field will be able to achieve in the next decade tend to run much
higher than what will likely be possible.

Twice in the past, AI went through a cycle of intense
optimism followed by disappointment and skepticism, with a dearth of funding as a
result. 

#### 1.1.8 The promise of AI

Although we may have unrealistic short-term expectations for AI, the long-term picture is looking bright. 

#### 1.2 Before deep learning: a brief history of machine learning

Deep learning is popular but not the only form of machine learning used.

####  1.2.1 Probabilistic modeling

Probabilistic modeling is the application of the principles of statistics to data analysis. It
was one of the earliest forms of machine learning, and it’s still widely used to this day.
One of the best-known algorithms in this category is the Naive Bayes algorithm.

 Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theorem while assuming that the features in the input data are all independent
 
A closely related model is the logistic regression (logreg for short), which is sometimes considered to be the “hello world” of modern machine learning. Don’t be misled by its name—logreg is a classification algorithm rather than a regression
algorithm.

#### 1.2.2 Early neural networks

. Although the core ideas of neural networks were investigated in toy forms as early
as the 1950s, the approach took decades to get started. For a long time, the missing piece
was an efficient way to train large neural networks. This changed in the mid-1980s,
Licensed to <null>
Before deep learning: a brief history of machine learning 15
when multiple people independently rediscovered the Backpropagation algorithm—
a way to train chains of parametric operations using gradient-descent optimization
(later in the book, we’ll precisely define these concepts)—and started applying it to
neural networks.

#### 1.2.3 Kernel methods

Kernel methods are a group of
classification algorithms, the best known of which is the support vector machine (SVM).

SVMs aim at solving classification problems by finding good
decision boundaries (see figure 1.10) between two sets of points
belonging to two different categories.

A kernel function is a computationally tractable operation that maps any
two points in your initial space to the distance between these points in your target
representation space, completely bypassing the explicit computation of the new representation.

#### 1.2.4 Decision trees, random forests, and gradient boosting machines

Decision trees are flowchart-like structures that let you classify input data points or predict output values given inputs (see figure 1.11). They’re easy to visualize and interpret.

the Random Forest algorithm introduced a robust, practical take on
decision-tree learning that involves building a large number of specialized decision
trees and then ensembling their outputs. 

A gradient boosting machine, much like a random forest, is a machine-learning
technique based on ensembling weak prediction models, generally decision trees. It
uses gradient boosting, a way to improve any machine-learning model by iteratively training new models that specialize in addressing the weak points of the previous models.

