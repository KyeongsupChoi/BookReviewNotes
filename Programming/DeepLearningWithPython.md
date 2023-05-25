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

- In 2016 and 2017, Kaggle was dominated by two approaches: gradient boosting machines and deep learning

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

- The key issue was that of gradient propagation through deep stacks of layers. The feedback signal used to train neural networks would fade away as the number of layers increased.

- The advent of several simple but important algorithmic improvements that allowed for better gradient propagation:
 Better activation functions for neural layers
 Better weight-initialization schemes, starting with layer-wise pretraining, which was
quickly abandoned
 Better optimization schemes, such as RMSProp and Adam

- Even more advanced ways to help gradient propagation were discovered, such as batch normalization, residual connections, and depthwise separable convolutions.

#### 1.3.4 A new wave of investment

- There was a gradual wave of industry investment far beyond anything previously seen in the history of AI

- The total venture capital investment in AI was around $19 million, which went almost entirely to practical applications of shallow machine-learning approaches. By 2014, it had risen to a staggering $394 million

- Machine learning—in particular, deep learning—has become central to the product strategy of these tech giants.

#### 1.3.5 The democratization of deep learning

- One of the key factors driving this inflow of new faces in deep learning has been the democratization of the toolsets used in the field. 

- This has been driven most notably by the development of Theano and then TensorFlow—two symbolic tensor-manipulation frameworks for Python that support autodifferentiation, greatly simplifying the implementation of new models—and by the rise of user-friendly libraries such as Keras, which makes deep learning as easy

#### 1.3.6 Will it last?

- (Simplicity)—Deep learning removes the need for feature engineering, replacing complex, brittle, engineering-heavy pipelines with simple, end-to-end trainable models that are typically built using only five or six different tensor operations

- (Scalability)—Deep learning is highly amenable to parallelization on GPUs or TPUs, so it can take full advantage of Moore’s law. In addition, deep-learning models are trained by iterating over small batches of data, allowing them to be trained on datasets of arbitrary size.

- (Versatility and reusability)—Unlike many prior machine-learning approaches, deep-learning models can be trained on additional data without restarting from scratch, making them viable for continuous online learning—an important property for very large production models.

### Chapter 2: Before we begin: the mathematical building blocks of neural networks  

- Understanding deep learning requires familiarity with many simple mathematical concepts: tensors, tensor operations, differentiation, gradient descent, and so on

#### 2.1 A first look at a neural network

- The core building block of neural networks is the layer, a data-processing module that you can think of as a filter for data

- Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand

- Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation

- Dense layers are densely connected (also called fully connected) neural layers.

- Softmax layer, which returns an array of 10 probability scores (summing to 1)
 
A Neural Network requires 3 components for training
 A (loss function)—How the network will be able to measure its performance on the training data, and thus how it will be able to steer itself in the right direction.
 An (optimizer)—The mechanism through which the network will update itself based on the data it sees and its loss function.
 (Metrics) to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the images that were correctly classified)

#### 2.2 Data representations for neural networks

- At its core, a tensor is a container for data—almost always numerical data. So, it’s a container for numbers. You may be already familiar with matrices, which are 2D tensors: tensors are a generalization of matrices to an arbitrary number of dimensions

#### 2.2.1 Scalars (0D tensors)

- A tensor that contains only one number is called a scalar (or scalar tensor, or 0-dimensional tensor, or 0D tensor). In Numpy, a float32 or float64 number is a scalar tensor (or scalar array).

#### 2.2.2 Vectors (1D tensors)

- An array of numbers is called a vector, or 1D tensor. A 1D tensor is said to have exactly one axis.

#### 2.2.3 Matrices (2D tensors)

- An array of vectors is a matrix, or 2D tensor. A matrix has two axes 

#### 2.2.4 3D tensors and higher-dimensional tensors

- By packing 3D tensors in an array, you can create a 4D tensor, and so on. In deep learning, you’ll generally manipulate tensors that are 0D to 4D, although you may go up to 5D if you process video data.

#### 2.2.5 Key attributes

 Number of axes (rank)—For instance, a 3D tensor has three axes, and a matrix has two axes. This is also called the tensor’s ndim in Python libraries such as Numpy.
 Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape(3, 5), and the 3D tensor example has shape (3, 3, 5). A vector has a shape with a single element, such as (5,), whereas a scalar has an empty shape, ().
 Data type (usually called dtype in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be float32, uint8, float64, and so on. On rare occasions, you may see a char tensor. Note that string tensors don’t exist in Numpy (or in most other libraries), because tensors live in preallocated, contiguous memory segments: and strings, being variable length, would preclude the use of this implementation.

#### 2.2.6 Manipulating tensors in Numpy

- Selecting specific elements in a tensor is called tensor slicing
 
#### 2.2.7 The notion of data batches

- In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis

- In addition, deep-learning models don’t process an entire dataset at once; rather, they break the data into small batches.

- When considering such a batch tensor, the first axis (axis 0) is called the batch axis or batch dimension. This is a term you’ll frequently encounter when using Keras and other deep-learning libraries

#### 2.2.8 Real-world examples of data tensors

 Vector data—2D tensors of shape (samples, features)
 Timeseries data or sequence data—3D tensors of shape (samples, timesteps, features)
 Images—4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
 Video—5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

#### 2.2.9 Vector data

- In the most common case, in the dataset, each single data point can be encoded as a vector, and thus a batch of data will be encoded as a 2D tensor (that is, an array of vectors), where the first axis is the samples axis and the second axis is the features axis.

#### 2.2.10 Timeseries data or sequence data

- Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a 3D tensor with an explicit time axis. Each sample can be encoded as a sequence of vectors (a 2D tensor), and thus a batch of data will be encoded as a 3D tensor

- The time axis is always the second axis (axis of index 1), by convention

#### 2.2.11 Image data

- Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in 2D tensors, by convention image tensors are always 3D, with a one dimensional color channel for grayscale images.

- There are two conventions for shapes of images tensors: the channels-last convention(used by TensorFlow) and the channels-first convention (used by Theano).

#### 2.2.12 Video data

- Video data is one of the few types of real-world data for which you’ll need 5D tensors.

- A video can be understood as a sequence of frames, each frame being a color image. Because each frame can be stored in a 3D tensor (height, width, color_depth), a sequence of frames can be stored in a 4D tensor (frames, height, width, color_depth), and thus a batch of different videos can be stored in a 5D tensor of shape

### 2.3 The gears of neural networks: tensor operations

- All transformations learned by deep neural networks can be reduced to a handful of tensor operations applied to tensors of numeric data.

#### 2.3.1 Element-wise operations

- Element-wise operations: operations that are applied independently to each entry in the tensors being considered. 

- This means these operations are highly amenable to massively parallel implementations 

- On the same principle, you can do element-wise multiplication, subtraction, and so on.

- In practice, when dealing with Numpy arrays, these operations are available as well-optimized built-in Numpy functions, which themselves delegate the heavy lifting to a Basic Linear Algebra Subprograms (BLAS) implementation if you have one installed(which you should). 

- BLAS are low-level, highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.

#### 2.3.2 Broadcasting

- When the shapes of the two tensors being added differ, when possible, and if there’s no ambiguity, the smaller tensor will be broadcasted to match the shape of the larger tensor

- 1 Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor.

- 2 The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.

#### 2.3.3 Tensor dot

- The dot operation, also called a tensor product (not to be confused with an elementwise product) is the most common, most useful tensor operation. Contrary to element-wise operations, it combines entries in the input tensors

- An element-wise product is done with the * operator in Numpy, Keras, Theano, and TensorFlow. dot uses a different syntax in TensorFlow, but in both Numpy and Keras it’s done using the standard dot operator:

#### 2.3.4 Tensor reshaping

- Reshaping a tensor means rearranging its rows and columns to match a target shape. Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor.

- A special case of reshaping that’s commonly encountered is transposition. Transposing a matrix means exchanging its rows and its columns,

#### 2.3.5 Geometric interpretation of tensor operations

- Because the contents of the tensors manipulated by tensor operations can be interpreted as coordinates of points in some geometric space, all tensor operations have a geometric interpretation.

- In general, elementary geometric operations such as affine transformations, rotations, scaling, and so on can be expressed as tensor operations.

#### 2.3.6 A geometric interpretation of deep learning

- You can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a long series of simple steps.

- Deep learning is finding neat representations for complex, highly folded data manifolds. 

- It takes the approach of incrementally decomposing a complicated geometric transformation into a long chain of elementary ones.

#### 2.4 The engine of neural networks:gradient-based optimization

- Weights or trainable parameters of the layer (the kernel and bias attributes, respectively). These weights contain the information learned by the network from exposure to training data.

- These weights must be gradually adjusted, based on a feedback signal. This gradual adjustment, also called training, is basically the learning that machine learning is all about.

- A training loop, which works as follows. Repeat these steps in a loop, as long as necessary:
1. Draw a batch of training samples x and corresponding targets y.
2. Run the network on x (a step called the forward pass) to obtain predictions y_pred.
3. Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y.
4. Update all weights of the network in a way that slightly reduces the loss on this batch.

- One approach is to take advantage of the fact that all operations used in the network are differentiable, and compute the gradient of the loss with regard to the network’s coefficients. 

- You can then move the coefficients in the opposite direction from the gradient, thus decreasing the loss.

#### 2.4.1 What’s a derivative?

-  the function is continuous, a small change in x can only result
in a small change in y—that’s the intuition behind continuity.

- The slope a is called the derivative of f in p.

#### 2.4.2 Derivative of a tensor operation: the gradient

- A gradient is the derivative of a tensor operation. It’s the generalization of the concept of derivatives to functions of multidimensional inputs: that is, to functions that take tensors as inputs

#### 2.4.3 Stochastic gradient descent

Given a differentiable function, it’s theoretically possible to find its minimum analytically: 

Applied to a neural network, that means finding analytically the combination of
weight values that yields the smallest possible loss function. 

This can be done by solving the equation gradient(f)(W) = 0 for W

it’s important to pick a reasonable value for the step factor.
If it’s too small, the descent down the curve will take many iterations, and it could get
stuck in a local minimum. If step is too large, your updates may end up taking you to
completely random locations on the curve

mini-batch stochastic gradient descent

there exist multiple variants of SGD that differ by taking into account
previous weight updates when computing the next weight update, rather than just
looking at the current value of the gradients

Momentum is implemented by moving
the ball at each step based not only on the current slope value (current acceleration)
but also on the current velocity (resulting from past acceleration). In practice, this
means updating the parameter w based not only on the current gradient value but also
on the previous parameter update, such as in this naive implementation

#### 2.4.4 Chaining derivatives: the Backpropagation algorithm

Calculus tells us that such a chain of functions can be derived using the following identity, called the chain rule: f(g(x)) = f'(g(x)) * g'(x). Applying the chain rule to the
computation of the gradient values of a neural network gives rise to an algorithm
called Backpropagation

Backpropagation starts with the final loss value and works backward from the top layers to the bottom layers, applying the chain rule to compute the contribution that each parameter
had in the loss value.

#### 2.5 Looking back at our first example

 this network consists of a chain of two Dense layers, that
each layer applies a few simple tensor operations to the input data, and that these
operations involve weight tensors.

## 3 Getting started with neural networks

This chapter is designed to get you started with using neural networks to solve real
problems

we’ll take a closer look at the core components of neural networks
that we introduced in chapter 2: layers, networks, objective functions, and optimizers.

We’ll give you a quick introduction to Keras, the Python deep-learning library
that we’ll use throughout the book

You’ll set up a deep-learning workstation, with TensorFlow, Keras, and GPU support

#### 3.1 Anatomy of a neural network

training a neural network revolves around the following objects:
 Layers, which are combined into a network (or model)
 The input data and corresponding targets
 The loss function, which defines the feedback signal used for learning
 The optimizer, which determines how learning proceeds

#### 3.1.1 Layers: the building blocks of deep learning

The fundamental data structure in neural networks is the layer

A layer is a data-processing module that takes as input one or
more tensors and that outputs one or more tensors.

simple vector data, stored in 2D tensors of shape (samples,
features), is often processed by densely connected layers, also called fully connected or dense
layers 

). Sequence data, stored in 3D tensors of shape (samples,
timesteps, features), is typically processed by recurrent layers such as an LSTM layer.

Image data, stored in 4D tensors, is usually processed by 2D convolution layers (Conv2D).

When using Keras, you don’t have to worry about
compatibility, because the layers you add to your models are dynamically built to
match the shape of the incoming layer

#### 3.1.2 Models: networks of layers

A deep-learning model is a directed, acyclic graph of layers. The most common
instance is a linear stack of layers, mapping a single input to a single output.

But as you move forward, you’ll be exposed to a much broader variety of network
topologies

 Some common ones include the following:
 Two-branch networks
 Multihead networks
 Inception blocks

By choosing a network topology, you constrain your space of possibilities
(hypothesis space) to a specific series of tensor operations, mapping input data to output data. 
What you’ll then be searching for is a good set of values for the weight tensors involved in these tensor operations.

Picking the right network architecture is more an art than a science; and although
there are some best practices and principles you can rely on, only practice can help
you become a proper neural-network architect.

#### 3.1.3 Loss functions and optimizers: keys to configuring the learning process

Loss function (objective function)—The quantity that will be minimized during
training. It represents a measure of success for the task at hand.

Optimizer—Determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD).

A neural network that has multiple outputs may have multiple loss functions (one per
output). But the gradient-descent process must be based on a single scalar loss value;
so, for multiloss networks, all losses are combined (via averaging) into a single scalar
quantity.

 Choosing the right objective function for the right problem is extremely important: your network will take any shortcut it can, to minimize the loss; so if the objective
doesn’t fully correlate with success for the task at hand, your network will end up
doing things you may not have wanted.

#### 3.2 Introduction to Keras

Keras is a
deep-learning framework for Python that provides a convenient way to define and
train almost any kind of deep-learning model.

 Keras has the following key features:
 It allows the same code to run seamlessly on CPU or GPU.
 It has a user-friendly API that makes it easy to quickly prototype deep-learning
models.
 It has built-in support for convolutional networks (for computer vision), recurrent networks (for sequence processing), and any combination of both.
 It supports arbitrary network architectures: multi-input or multi-output models,
layer sharing, model sharing, and so on. This means Keras is appropriate for
building essentially any deep-learning model, from a generative adversarial network to a neural Turing machine.

#### 3.2.1 Keras, TensorFlow, Theano, and CNTK

Keras is a model-level library, providing high-level building blocks for developing
deep-learning models. It doesn’t handle low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized, well-optimized tensor
library to do so, serving as the backend engine of Keras. 

s. Rather than choosing a single
tensor library and tying the implementation of Keras to that library, Keras handles the
problem in a modular way (see figure 3.3); thus several different backend engines can
be plugged seamlessly into Keras.

. Currently, the three existing backend implementations are the TensorFlow backend, the Theano backend, and the Microsoft Cognitive
Toolkit (CNTK) backend. In the future, it’s likely that Keras will be extended to work
with even more deep-learning execution engines.

We recommend using the TensorFlow backend as
the default for most of your deep-learning needs, because it’s the most widely adopted,
scalable, and production ready.

Via TensorFlow (or Theano, or CNTK), Keras is able to run seamlessly on both
CPUs and GPUs. When running on CPU, TensorFlow is itself wrapping a low-level
library for tensor operations called Eigen (http://eigen.tuxfamily.org). On GPU,
TensorFlow wraps a library of well-optimized deep-learning operations called the
NVIDIA CUDA Deep Neural Network library

#### 3.2.2 Developing with Keras: a quick overview

The typical
Keras workflow looks just like that example:
1. Define your training data: input tensors and target tensors.
2. Define a network of layers (or model ) that maps your inputs to your targets.
3. Configure the learning process by choosing a loss function, an optimizer, and
some metrics to monitor.
4. Iterate on your training data by calling the fit() method of your model.

With the functional API, you’re manipulating the data tensors that the model processes and applying layers to this tensor as if they were functions.

Finally, the learning process consists of passing Numpy arrays of input data (and the
corresponding target data) to the model via the fit() method, similar to what you
would do in Scikit-Learn and several other machine-learning libraries:

### 3.3 Setting up a deep-learning workstation

It’s highly recommended, although not strictly necessary, that you
run deep-learning code on a modern NVIDIA GPU. Some applications—in particular,
image processing with convolutional networks and sequence processing with recurrent neural networks—will be excruciatingly slow on CPU, even a fast multicore CPU.

Whether you’re running locally or in the cloud, it’s better to be using a Unix workstation. Although it’s technically possible to use Keras on Windows (all three Keras
backends support Windows), We don’t recommend it. 

#### 3.3.1 Jupyter notebooks: the preferred way to run deep-learning experiments

Jupyter notebooks are a great way to run deep-learning experiments—in particular,
the many code examples in this book. They’re widely used in the data-science and
machine-learning communities.

A notebook is a file generated by the Jupyter Notebook
app, which you can edit in your browser. It mixes the ability to
execute Python code with rich text-editing capabilities for annotating what you’re
doing. 

#### 3.3.2 Getting Keras running: two options

Use the official EC2 Deep Learning AMI (https://aws.amazon.com/amazonai/amis), and run Keras experiments as Jupyter notebooks on EC2. Do this if
you don’t already have a GPU on your local machine.

 Install everything from scratch on a local Unix workstation. You can then run
either local Jupyter notebooks or a regular Python codebase. Do this if you
already have a high-end NVIDIA GPU.

#### 3.3.3 Running deep-learning jobs in the cloud: pros and cons

If you don’t already have a GPU that you can use for deep learning (a recent, high-end
NVIDIA GPU), then running deep-learning experiments in the cloud is a simple, lowcost way for you to get started without having to buy any additional hardware. 

But if you’re a heavy user of deep learning, this setup isn’t sustainable in the long
term—or even for more than a few weeks. EC2 instances are expensive

Meanwhile, a solid consumerclass GPU will cost you somewhere between $1,000 and $1,500—a price that has been
fairly stable over time, even as the specs of these GPUs keep improving.

#### 3.3.4 What is the best GPU for deep learning?

A pretty powerful GPU

### 3.4 Classifying movie reviews: a binary classification example

Two-class classification, or binary classification, may be the most widely applied kind
of machine-learning problem

#### 3.4.1 The IMDB dataset

the IMDB dataset: a set of 50,000 highly polarized reviews from the
Internet Movie Database. They’re split into 25,000 reviews for training and 25,000
reviews for testing, each set consisting of 50% negative and 50% positive reviews.

it’s possible that your model could end up merely memorizing a mapping between your training samples and their targets, which would be
useless for the task of predicting targets for data the model has never seen before.

the IMDB dataset comes packaged with Keras. It has
already been preprocessed: the reviews (sequences of words) have been turned into
sequences of integers, where each integer stands for a specific word in a dictionary.

The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows
you to work with vector data of manageable size.

Because you’re restricting yourself to the top 10,000 most frequent words, no word
index will exceed 10,000

#### 3.4.2 Preparing the data

You can’t feed lists of integers into a neural network. You have to turn your lists into
tensors. There are two ways to do that:

 Pad your lists so that they all have the same length, turn them into an integer
tensor of shape (samples, word_indices), and then use as the first layer in
your network a layer capable of handling such integer tensors

 One-hot encode your lists to turn them into vectors of 0s and 1s. This would
mean, for instance, turning the sequence [3, 5] into a 10,000-dimensional vector that would be all 0s except for indices 3 and 5, which would be 1s. Then you
could use as the first layer in your network a Dense layer, capable of handling
floating-point vector data.

#### 3.4.3 Building your network

The input data is vectors, and the labels are scalars (1s and 0s): this is the easiest setup
you’ll ever encounter. A type of network that performs well on such a problem is
a simple stack of fully connected (Dense) layers with relu activations: Dense(16,
activation='relu').

 The argument being passed to each Dense layer (16) is the number of hidden
units of the layer. A hidden unit is a dimension in the representation space of the layer.

Having 16 hidden units means the weight matrix W will have shape (input_dimension,
16): the dot product with W will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector b and apply the relu operation). 

Having more hidden units (a higher-dimensional representation space)
allows your network to learn more-complex representations, but it makes the network
more computationally expensive and may lead to learning unwanted patterns

There are two key architecture decisions to be made about such a stack of Dense layers:
 How many layers to use
 How many hidden units to choose for each layer

Because you’re facing a
binary classification problem and the output of your network is a probability (you end
your network with a single-unit layer with a sigmoid activation), it’s best to use the
binary_crossentropy loss. It isn’t the only viable choice: you could use, for instance,
mean_squared_error. But crossentropy is usually the best choice when you’re dealing
with models that output probabilities.

 Crossentropy is a quantity from the field of Information Theory that measures the distance between probability distributions or, in this
case, between the ground-truth distribution and your predictions.

#### 3.4.4 Validating your approach

In order to monitor during training the accuracy of the model on data it has never
seen before, you’ll create a validation set by setting apart 10,000 samples from the
original training data.

 the training loss decreases with every epoch, and the training accuracy
increases with every epoch. That’s what you would expect when running gradientdescent optimization—the quantity you’re trying to minimize should be less with
every iteration. 

But that isn’t the case for the validation loss and accuracy: they seem to
peak at the fourth epoch. This is an example of what we warned against earlier: a
model that performs better on the training data isn’t necessarily a model that will do
better on data it has never seen before. In precise terms, what you’re seeing is overfitting: 

This fairly naive approach achieves an accuracy of 88%. With state-of-the-art
approaches, you should be able to get close to 95%.

#### 3.4.5 Using a trained network to generate predictions on new data

You can generate the likelihood of reviews being positive by using the predict method:

#### 3.4.6 Further experiments

The following experiments will help convince you that the architecture choices you’ve
made are all fairly reasonable, although there’s still room for improvement:
 You used two hidden layers. Try using one or three hidden layers, and see how
doing so affects validation and test accuracy.
 Try using layers with more hidden units or fewer hidden units: 32 units, 64 units,
and so on.
 Try using the mse loss function instead of binary_crossentropy.
 Try using the tanh activation (an activation that was popular in the early days of
neural networks) instead of relu.

#### 3.4.7 Wrapping up

Here’s what you should take away from this example:
 You usually need to do quite a bit of preprocessing on your raw data in order to
be able to feed it—as tensors—into a neural network. Sequences of words can
be encoded as binary vectors, but there are other encoding options, too.
 Stacks of Dense layers with relu activations can solve a wide range of problems
(including sentiment classification), and you’ll likely use them frequently.
 In a binary classification problem (two output classes), your network should
end with a Dense layer with one unit and a sigmoid activation: the output of
your network should be a scalar between 0 and 1, encoding a probability.
 With such a scalar sigmoid output on a binary classification problem, the loss
function you should use is binary_crossentropy.
 The rmsprop optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.
 As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never
seen before. Be sure to always monitor performance on data that is outside of
the training set. 

