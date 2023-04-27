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

- In particular, deep learning has achieved the following breakthroughs, all in historically difficult areas of machine learning:
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

- Although deep learning has led to remarkable achievements in recent years, expectations for what the field will be able to achieve in the next decade tend to run much higher than what will likely be possible.

- Twice in the past, AI went through a cycle of intense optimism followed by disappointment and skepticism, with a dearth of funding as a result. 

#### 1.1.8 The promise of AI

- Although we may have unrealistic short-term expectations for AI, the long-term picture is looking bright. 

#### 1.2 Before deep learning: a brief history of machine learning

- Deep learning is popular but not the only form of machine learning used.

####  1.2.1 Probabilistic modeling

- Probabilistic modeling is the application of the principles of statistics to data analysis. It was one of the earliest forms of machine learning, and it’s still widely used to this day. One of the best-known algorithms in this category is the Naive Bayes algorithm.

- Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theorem while assuming that the features in the input data are all independent
 
- A closely related model is the logistic regression (logreg for short), which is sometimes considered to be the “hello world” of modern machine learning. Don’t be misled by its name—logreg is a classification algorithm rather than a regression algorithm.

#### 1.2.2 Early neural networks
 
- Although the core ideas of neural networks were investigated in toy forms as early as the 1950s, the approach took decades to get started. For a long time, the missing piece was an efficient way to train large neural networks.

- Multiple people independently rediscovered the Backpropagation algorithm a way to train chains of parametric operations using gradient-descent optimization and started applying it to neural networks.

#### 1.2.3 Kernel methods

- Kernel methods are a group of classification algorithms, the best known of which is the support vector machine (SVM).

- SVMs aim at solving classification problems by finding good decision boundaries (see figure 1.10) between two sets of points belonging to two different categories.

- A kernel function is a computationally tractable operation that maps any two points in your initial space to the distance between these points in your target representation space, completely bypassing the explicit computation of the new representation.

#### 1.2.4 Decision trees, random forests, and gradient boosting machines

- Decision trees are flowchart-like structures that let you classify input data points or predict output values given inputs (see figure 1.11). They’re easy to visualize and interpret.

- The Random Forest algorithm introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their outputs. 

- A gradient boosting machine, much like a random forest, is a machine-learning  technique based on ensembling weak prediction models, generally decision trees. It uses gradient boosting, a way to improve any machine-learning model by iteratively training new models that specialize in addressing the weak points of the previous models.

#### 1.2.5 Back to neural networks

- Since 2012, deep convolutional neural networks (convnets) have become the go-to algorithm for all computer vision tasks;

- Deep learning has also found applications in many other types of problems, such as natural-language processing.

#### 1.2.6 What makes deep learning different

- The primary reason deep learning took off so quickly is that it offered better performance on many problems.

- Deep learning also makes problem-solving much easier, because it completely automates what used to be the most crucial step in a machine-learning workflow: feature engineering

- There are fast-diminishing returns to successive applications of shallow-learning methods, because the optimal first representation layer in a three layer model isn’t the optimal first layer in a one-layer or two-layer model. 

- What is transformative about deep learning is that it allows a model to learn all layers of representation jointly, at the same time, rather than in succession

- These are the two essential characteristics of how deep learning learns from data the incremental, layer-by-layer way in which increasingly complex representations are developed, and the fact that these intermediate incremental representations are learned jointly

#### 1.2.7 The modern machine-learning landscape

- A great way to get a sense of the current landscape of machine-learning algorithms and tools is to look at machine-learning competitions on Kaggle. 

-In 2016 and 2017, Kaggle was dominated by two approaches: gradient boosting machines and deep learning

#### 1.3 Why deep learning? Why now?

- In general, three technical forces are driving advances in machine learning:
 Hardware
 Datasets and benchmarks
 Algorithmic advances

#### 1.3.1 Hardware

- Between 1990 and 2010, off-the-shelf CPUs became faster by a factor of approximately 5,000.

- Throughout the 2000s, companies like NVIDIA and AMD have been investing billions of dollars in developing fast, massively parallel chips (graphical processing units, GPUs) to power the graphics of increasingly photorealistic video games— cheap, single-purpose supercomputers designed to render complex 3D scenes on yourscreen in real time. This investment came to benefit the scientific community when, in 2007, NVIDIA launched CUDA, a programming interface for its line of GPUs. 

- The deep-learning industry is starting to go beyond GPUs and is investing in increasingly specialized, efficient chips for deep learning.

#### 1.3.2 Data

- The game changer has been the rise of the internet, making it feasible to collect and distribute very large datasets for machine learning

- If there’s one dataset that has been a catalyst for the rise of deep learning, it’s the ImageNet dataset, consisting of 1.4 million images that have been hand annotated with 1,000 image categories

#### 1.3.3 Algorithms

The key issue was that
of gradient propagation through deep stacks of layers. The feedback signal used to train
neural networks would fade away as the number of layers increased.

the advent of several simple but important
algorithmic improvements that allowed for better gradient propagation:
 Better activation functions for neural layers
 Better weight-initialization schemes, starting with layer-wise pretraining, which was
quickly abandoned
 Better optimization schemes, such as RMSProp and Adam

even more advanced ways to help gradient propagation were discovered, such as batch normalization, residual connections, and depthwise separable convolutions.

#### 1.3.4 A new wave of investment

What followed was
a gradual wave of industry investment far beyond anything previously seen in the history of AI

the total venture capital
investment in AI was around $19 million, which went almost entirely to practical applications of shallow machine-learning approaches. By 2014, it had risen to a staggering
$394 million

Machine learning—in particular, deep learning—has become central to the product strategy of these tech giants.

#### 1.3.5 The democratization of deep learning

One of the key factors driving this inflow of new faces in deep learning has been the
democratization of the toolsets used in the field. 

This has been
driven most notably by the development of Theano and then TensorFlow—two symbolic
tensor-manipulation frameworks for Python that support autodifferentiation, greatly simplifying the implementation of new models—and by the rise of user-friendly libraries
such as Keras, which makes deep learning as easy

#### 1.3.6 Will it last?

Simplicity—Deep learning removes the need for feature engineering, replacing
complex, brittle, engineering-heavy pipelines with simple, end-to-end trainable
models that are typically built using only five or six different tensor operations

Scalability—Deep learning is highly amenable to parallelization on GPUs or
TPUs, so it can take full advantage of Moore’s law. In addition, deep-learning
models are trained by iterating over small batches of data, allowing them to be
trained on datasets of arbitrary size.

Versatility and reusability—Unlike many prior machine-learning approaches,
deep-learning models can be trained on additional data without restarting from
scratch, making them viable for continuous online learning—an important
property for very large production models.

### Chapter 2: Before we begin: the mathematical building blocks of neural networks  

Understanding deep learning requires familiarity with many simple mathematical
concepts: tensors, tensor operations, differentiation, gradient descent, and so on

