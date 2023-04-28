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

#### 2.1 A first look at a neural network

The core building block of neural networks is the layer, a data-processing module that
you can think of as a filter for data

Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand

Most of
deep learning consists of chaining together simple layers that will implement a form
of progressive data distillation

 Dense layers, which are densely
connected (also called fully connected) neural layers. T

 softmax layer, which means it will return an array of 10 probability scores (summing to 1)
 
 A loss function—How the network will be able to measure its performance on
the training data, and thus how it will be able to steer itself in the right direction.
 An optimizer—The mechanism through which the network will update itself
based on the data it sees and its loss function.
 Metrics to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the images that were correctly classified)

#### 2.2 Data representations for neural networks

 At its core, a tensor is a container for data—almost always numerical data. So, it’s a
container for numbers. You may be already familiar with matrices, which are 2D tensors: tensors are a generalization of matrices to an arbitrary number of dimensions

#### 2.2.1 Scalars (0D tensors)

A tensor that contains only one number is called a scalar (or scalar tensor, or 0-dimensional
tensor, or 0D tensor). In Numpy, a float32 or float64 number is a scalar tensor (or scalar
array).

#### 2.2.2 Vectors (1D tensors)

An array of numbers is called a vector, or 1D tensor. A 1D tensor is said to have exactly
one axis.

#### 2.2.3 Matrices (2D tensors)

An array of vectors is a matrix, or 2D tensor. A matrix has two axes 

#### 2.2.4 3D tensors and higher-dimensional tensors

By packing 3D tensors in an array, you can create a 4D tensor, and so on. In deep learning, you’ll generally manipulate tensors that are 0D to 4D, although you may go up to
5D if you process video data.

#### 2.2.5 Key attributes

 Number of axes (rank)—For instance, a 3D tensor has three axes, and a matrix has
two axes. This is also called the tensor’s ndim in Python libraries such as Numpy.
 Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape
(3, 5), and the 3D tensor example has shape (3, 3, 5). A vector has a shape
with a single element, such as (5,), whereas a scalar has an empty shape, ().
 Data type (usually called dtype in Python libraries)—This is the type of the data
contained in the tensor; for instance, a tensor’s type could be float32, uint8,
float64, and so on. On rare occasions, you may see a char tensor. Note that
string tensors don’t exist in Numpy (or in most other libraries), because tensors
live in preallocated, contiguous memory segments: and strings, being variable
length, would preclude the use of this implementation.

#### 2.2.6 Manipulating tensors in Numpy

 Selecting specific elements in a tensor is called tensor slicing
 
#### 2.2.7 The notion of data batches

In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll
come across in deep learning will be the samples axis

 In addition, deep-learning models don’t process an entire dataset at once; rather,
they break the data into small batches.

When considering such a batch tensor, the first axis (axis 0) is called the batch axis or
batch dimension. This is a term you’ll frequently encounter when using Keras and other
deep-learning libraries

#### 2.2.8 Real-world examples of data tensors

 Vector data—2D tensors of shape (samples, features)
 Timeseries data or sequence data—3D tensors of shape (samples, timesteps,
features)
 Images—4D tensors of shape (samples, height, width, channels) or (samples,
channels, height, width)
 Video—5D tensors of shape (samples, frames, height, width, channels) or
(samples, frames, channels, height, width)

#### 2.2.9 Vector data

This is the most common case. In such a dataset, each single data point can be encoded
as a vector, and thus a batch of data will be encoded as a 2D tensor (that is, an array of
vectors), where the first axis is the samples axis and the second axis is the features axis.

#### 2.2.10 Timeseries data or sequence data

Whenever time matters in your data (or the notion of sequence order), it makes sense
to store it in a 3D tensor with an explicit time axis. Each sample can be encoded as a
sequence of vectors (a 2D tensor), and thus a batch of data will be encoded as a 3D
tensor

The time axis is always the second axis (axis of index 1), by convention

#### 2.2.11 Image data

Images typically have three dimensions: height, width, and color depth. Although
grayscale images (like our MNIST digits) have only a single color channel and could
thus be stored in 2D tensors, by convention image tensors are always 3D, with a onedimensional color channel for grayscale images.

There are two conventions for shapes of images tensors: the channels-last convention
(used by TensorFlow) and the channels-first convention (used by Theano).

#### 2.2.12 Video data

Video data is one of the few types of real-world data for which you’ll need 5D tensors.

A video can be understood as a sequence of frames, each frame being a color image.
Because each frame can be stored in a 3D tensor (height, width, color_depth), a
sequence of frames can be stored in a 4D tensor (frames, height, width, color_
depth), and thus a batch of different videos can be stored in a 5D tensor of shape

### 2.3 The gears of neural networks: tensor operations

 all transformations learned
by deep neural networks can be reduced to a handful of tensor operations applied to
tensors of numeric data.

#### 2.3.1 Element-wise operations

element-wise operations: operations that are
applied independently to each entry in the tensors being considered. 

This means
these operations are highly amenable to massively parallel implementations 

On the same principle, you can do element-wise multiplication, subtraction, and so on.

In practice, when dealing with Numpy arrays, these operations are available as welloptimized built-in Numpy functions, which themselves delegate the heavy lifting to a
Basic Linear Algebra Subprograms (BLAS) implementation if you have one installed
(which you should). BLAS are low-level, highly parallel, efficient tensor-manipulation
routines that are typically implemented in Fortran or C.

#### 2.3.2 Broadcasting

What happens with addition when the shapes of the two tensors
being added differ?
 When possible, and if there’s no ambiguity, the smaller tensor will be broadcasted to
match the shape of the larger tensor

1 Axes (called broadcast axes) are added to the smaller tensor to match the ndim of
the larger tensor.
2 The smaller tensor is repeated alongside these new axes to match the full shape
of the larger tensor.

#### 2.3.3 Tensor dot

The dot operation, also called a tensor product (not to be confused with an elementwise product) is the most common, most useful tensor operation. Contrary to
element-wise operations, it combines entries in the input tensors

 An element-wise product is done with the * operator in Numpy, Keras, Theano,
and TensorFlow. dot uses a different syntax in TensorFlow, but in both Numpy and
Keras it’s done using the standard dot operator:

#### 2.3.4 Tensor reshaping

Reshaping a tensor means rearranging its rows and columns to match a target shape.
Naturally, the reshaped tensor has the same total number of coefficients as the initial
tensor.

A special case of reshaping that’s commonly encountered is transposition. Transposing a
matrix means exchanging its rows and its columns,

#### 2.3.5 Geometric interpretation of tensor operations

Because the contents of the tensors manipulated by tensor operations can be interpreted as coordinates of points in some geometric space, all tensor operations have a
geometric interpretation.

In general, elementary geometric operations such as affine transformations, rotations,
scaling, and so on can be expressed as tensor operations.

#### 2.3.6 A geometric interpretation of deep learning

you can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a long series of simple steps.

finding neat representations for complex, highly folded data manifolds. 

it takes the approach of
incrementally decomposing a complicated geometric transformation into a long
chain of elementary ones,