# Craftsmen Information Processing

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

##### 1.1.1 Artificial Intelligence 

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

