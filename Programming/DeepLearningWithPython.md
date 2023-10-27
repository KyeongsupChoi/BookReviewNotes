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

- Given a differentiable function, it’s theoretically possible to find its minimum analytically: 

- Applied to a neural network, that means finding analytically the combination of weight values that yields the smallest possible loss function. 

- This can be done by solving the equation gradient(f)(W) = 0 for W

- It’s important to pick a reasonable value for the step factor. If it’s too small, the descent down the curve will take many iterations, and it could get stuck in a local minimum. If step is too large, your updates may end up taking you to completely random locations on the curve

- Mini-batch stochastic gradient descent

- There exist multiple variants of SGD that differ by taking into account previous weight updates when computing the next weight update, rather than just looking at the current value of the gradients

- Momentum is implemented by moving the ball at each step based not only on the current slope value (current acceleration)but also on the current velocity (resulting from past acceleration). In practice, this means updating the parameter w based not only on the current gradient value but also on the previous parameter update, such as in this naive implementation

#### 2.4.4 Chaining derivatives: the Backpropagation algorithm

- Calculus tells us that such a chain of functions can be derived using the following identity, called the chain rule: f(g(x)) = f'(g(x)) * g'(x). Applying the chain rule to the computation of the gradient values of a neural network gives rise to an algorithm called Backpropagation

- Backpropagation starts with the final loss value and works backward from the top layers to the bottom layers, applying the chain rule to compute the contribution that each parameter had in the loss value.

#### 2.5 Looking back at our first example

- This network consists of a chain of two Dense layers, that each layer applies a few simple tensor operations to the input data, and that these operations involve weight tensors.

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

#### 3.5 Classifying newswires: a multiclass classification example

you’ll build a network to classify Reuters newswires into 46 mutually
exclusive topics. Because you have many classes, this problem is an instance of multiclass classification; and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification.

If each data point could belong to multiple categories (in this case, topics), you’d be
facing a multilabel, multiclass classification problem

#### 3.5.1 The Reuters dataset

You’ll work with the Reuters dataset, a set of short newswires and their topics, published
by Reuters in 1986. It’s a simple, widely used toy dataset for text classification. There
are 46 different topics; some topics are more represented than others, but each topic
has at least 10 examples in the training set.

As with the IMDB dataset, the argument num_words=10000 restricts the data to the
10,000 most frequently occurring words found in the data.

#### 3.5.2 Preparing the data

To vectorize the labels, there are two possibilities: you can cast the label list as an integer tensor, or you can use one-hot encoding.

One-hot encoding is a widely used format for categorical data, also called categorical encoding

#### 3.5.3 Building your network

In a stack of Dense layers like that you’ve been using, each layer can only access information present in the output of the previous layer. If one layer drops some information
relevant to the classification problem, this information can never be recovered by later
layers: each layer can potentially become an information bottleneck.

The best loss function to use in this case is categorical_crossentropy. It measures
the distance between two probability distributions: here, between the probability distribution output by the network and the true distribution of the labels. By minimizing
the distance between these two distributions, you train the network to output something as close as possible to the true labels.

#### 3.5.4 Validating your approach

 let’s train the network for 20 epochs.
 
The network begins to overfit after nine epochs

This approach reaches an accuracy of ~80%. With a balanced binary classification
problem, the accuracy reached by a purely random classifier would be 50%. But in
this case it’s closer to 19%, so the results seem pretty good, at least when compared to
a random baseline:

#### 3.5.5 Generating predictions on new data

You can verify that the predict method of the model instance returns a probability
distribution over all 46 topics. Let’s generate topic predictions for all of the test data

#### 3.5.6 A different way to handle the labels and the loss

The loss
function used in listing 3.21, categorical_crossentropy, expects the labels to follow
a categorical encoding. With integer labels, you should use sparse_categorical_

This new loss function is still mathematically the same as categorical_crossentropy;
it just has a different interface.

#### 3.5.7 The importance of having sufficiently large intermediate layers

We mentioned earlier that because the final outputs are 46-dimensional, you should
avoid intermediate layers with many fewer than 46 hidden units. Now let’s see what
happens when you introduce an information bottleneck by having intermediate layers
that are significantly less than 46-dimensional

The network now peaks at ~71% validation accuracy, an 8% absolute drop. This drop
is mostly due to the fact that you’re trying to compress a lot of information (enough
information to recover the separation hyperplanes of 46 classes) into an intermediate
space that is too low-dimensional. The network is able to cram most of the necessary
information into these eight-dimensional representations, but not all of it

#### 3.5.8 Further experiments

 Try using larger or smaller layers: 32 units, 128 units, and so on.
 You used two hidden layers. Now try using a single hidden layer, or three hidden layers.

#### 3.5.9 Wrapping up

Here’s what you should take away from this example:
 If you’re trying to classify data points among N classes, your network should end
with a Dense layer of size N.
 In a single-label, multiclass classification problem, your network should end
with a softmax activation so that it will output a probability distribution over the
N output classes.
 Categorical crossentropy is almost always the loss function you should use for
such problems. It minimizes the distance between the probability distributions
output by the network and the true distribution of the targets.
 There are two ways to handle labels in multiclass classification:
– Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function
– Encoding the labels as integers and using the sparse_categorical_crossentropy
loss function
 If you need to classify data into a large number of categories, you should avoid
creating information bottlenecks in your network due to intermediate layers
that are too small.

### 3.6 Predicting house prices: a regression example

The two previous examples were considered classification problems, where the goal
was to predict a single discrete label of an input data point. Another common type of
machine-learning problem is regression, which consists of predicting a continuous
value instead of a discrete label:

#### 3.6.1 The Boston Housing Price dataset

You’ll attempt to predict the median price of homes in a given Boston suburb in the
mid-1970s, given data points about the suburb at the time, such as the crime rate, the
local property tax rate, and so on. The dataset you’ll use has an interesting difference
from the two previous examples. It has relatively few data points: only 506, split
between 404 training samples and 102 test samples. And each feature in the input data
(for example, the crime rate) has a different scale

#### 3.6.2 Preparing the data

It would be problematic to feed into a neural network values that all take wildly different ranges. The network might be able to automatically adapt to such heterogeneous
data, but it would definitely make learning more difficult. A widespread best practice
to deal with such data is to do feature-wise normalization: for each feature in the input
data (a column in the input data matrix), you subtract the mean of the feature and
divide by the standard deviation, so that the feature is centered around 0 and has a
unit standard deviation. This is easily done in Numpy.

Note that the quantities used for normalizing the test data are computed using the
training data. You should never use in your workflow any quantity computed on the
test data, even for something as simple as data normalization.

#### 3.6.3 Building your network

In general, the less training data you have, the worse overfitting will be, and using a small network is one way to mitigate overfitting

The network ends with a single unit and no activation (it will be a linear layer). This is
a typical setup for scalar regression (a regression where you’re trying to predict a single
continuous value). Applying an activation function would constrain the range the output can take; for instance, if you applied a sigmoid activation function to the last layer,
the network could only learn to predict values between 0 and 1. Here, because the last
layer is purely linear, the network is free to learn to predict values in any range.

Note that you compile the network with the mse loss function—mean squared error,
the square of the difference between the predictions and the targets. This is a widely
used loss function for regression problems.

You’re also monitoring a new metric during training: mean absolute error (MAE). It’s
the absolute value of the difference between the predictions and the targets. For
instance, an MAE of 0.5 on this problem would mean your predictions are off by $500
on average.

#### 3.6.4 Validating your approach using K-fold validation

To evaluate your network while you keep adjusting its parameters (such as the number
of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points,
the validation set would end up being very small (for instance, about 100 examples).
As a consequence, the validation scores might change a lot depending on which data
points you chose to use for validation and which you chose for training: the validation
scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 3.11).
It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on
the remaining partition. The validation score for the model used is then the average of
the K validation scores obtained.

The different runs do indeed show rather different validation scores, from 2.6 to 3.2.
The average (3.0) is a much more reliable metric than any single score—that’s the
entire point of K-fold cross-validation.

It may be a little difficult to see the plot, due to scaling issues and relatively high variance. Let’s do the following:
 Omit the first 10 data points, which are on a different scale than the rest of the curve.
 Replace each point with an exponential moving average of the previous points,
to obtain a smooth curve.

According to this plot, validation MAE stops improving significantly after 80 epochs.
Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the
number of epochs, you could also adjust the size of the hidden layers), you can train a
final production model on all of the training data, with the best parameters, and then
look at its performance on the test data.

#### 3.6.5 Wrapping up

 Regression is done using different loss functions than what we used for classification. Mean squared error (MSE) is a loss function commonly used for regression.
 Similarly, evaluation metrics to be used for regression differ from those used for
classification; naturally, the concept of accuracy doesn’t apply for regression. A
common regression metric is mean absolute error (MAE).
 When features in the input data have values in different ranges, each feature
should be scaled independently as a preprocessing step.
 When there is little data available, using K-fold validation is a great way to reliably evaluate a model.
 When little training data is available, it’s preferable to use a small network with
few hidden layers (typically only one or two), in order to avoid severe overfitting. 

## 4 Fundamentals of machine learning

This chapter will formalize some of your new intuition into a solid conceptual
framework for attacking and solving deep-learning problems. We’ll consolidate all
of these concepts—model evaluation, data preprocessing and feature engineering,
and tackling overfitting—into a detailed seven-step workflow for tackling any
machine-learning task.

### 4.1 Four branches of machine learning

binary classification, multiclass classification, and scalar
regression. All three are instances of supervised learning, where the goal is to learn the
relationship between training inputs and training targets.

Supervised learning is just the tip of the iceberg—machine learning is a vast field
with a complex subfield taxonomy. Machine-learning algorithms generally fall into
four broad categories, described in the following sections

#### 4.1.1 Supervised learning

 It consists of learning to map input data to
known targets (also called annotations), given a set of examples (often annotated by
humans).

Generally, almost all applications of deep learning
that are in the spotlight these days belong in this category, such as optical character
recognition, speech recognition, image classification, and language translation.

 Sequence generation—Given a picture, predict a caption describing it. Sequence
generation can sometimes be reformulated as a series of classification problems
(such as repeatedly predicting a word or token in a sequence).
 Syntax tree prediction—Given a sentence, predict its decomposition into a syntax
tree.
 Object detection—Given a picture, draw a bounding box around certain objects
inside the picture. This can also be expressed as a classification problem (given
many candidate bounding boxes, classify the contents of each one) or as a joint
classification and regression problem, where the bounding-box coordinates are
predicted via vector regression.
 Image segmentation—Given a picture, draw a pixel-level mask on a specific object. 

#### 4.1.2 Unsupervised learning

This branch of machine learning consists of finding interesting transformations of the
input data without the help of any targets, for the purposes of data visualization, data
compression, or data denoising, or to better understand the correlations present in
the data at hand. Unsupervised learning is the bread and butter of data analytics, and
it’s often a necessary step in better understanding a dataset before attempting to solve
a supervised-learning problem. Dimensionality reduction and clustering are well-known
categories of unsupervised learning. 

#### 4.1.3 Self-supervised learning

This is a specific instance of supervised learning, but it’s different enough that it
deserves its own category. Self-supervised learning is supervised learning without human-annotated labels—you can think of it as supervised learning without any
humans in the loop. There are still labels involved (because the learning has to be
supervised by something), but they’re generated from the input data, typically using a
heuristic algorithm.

 For instance, autoencoders are a well-known instance of self-supervised learning,
where the generated targets are the input, unmodified. In the same way, trying to predict the next frame in a video, given past frames, or the next word in a text, given previous words, are instances of self-supervised learning (temporally supervised learning, in this
case: supervision comes from future input data). Note that the distinction between
supervised, self-supervised, and unsupervised learning can be blurry sometimes—these
categories are more of a continuum without solid borders. Self-supervised learning can
be reinterpreted as either supervised or unsupervised learning, depending on whether
you pay attention to the learning mechanism or to the context of its application.

#### 4.1.4 Reinforcement learning

In reinforcement learning,
an agent receives information about its environment and learns to choose actions that
will maximize some reward. For instance, a neural network that “looks” at a videogame screen and outputs game actions in order to maximize its score can be trained
via reinforcement learning.

 In time, however, we expect to see reinforcement learning take over an increasingly large range of real-world applications:
self-driving cars, robotics, resource management, education, and so on. 

### 4.2 Evaluating machine-learning models

In the three examples presented in chapter 3, we split the data into a training set, a
validation set, and a test set. The reason not to evaluate the models on the same data
they were trained on quickly became evident: after just a few epochs, all three models
began to overfit. That is, their performance on never-before-seen data started stalling
(or worsening) compared to their performance on the training data—which always
improves as training progresses.

 In machine learning, the goal is to achieve models that generalize—that perform
well on never-before-seen data—and overfitting is the central obstacle. You can only
control that which you can observe, so it’s crucial to be able to reliably measure the
generalization power of your model. The following sections look at strategies for mitigating overfitting and maximizing generalization. In this section, we’ll focus on how
to measure generalization: how to evaluate machine-learning models.

#### 4.2.1 Training, validation, and test sets

Evaluating a model always boils down to splitting the available data into three sets:
training, validation, and test. You train on the training data and evaluate your model
on the validation data. Once your model is ready for prime time, you test it one final
time on the test data

The reason is that developing a model always involves tuning its configuration: for
example, choosing the number of layers or the size of the layers (called the hyperparameters of the model, to distinguish them from the parameters, which are the network’s weights). You do this tuning by using as a feedback signal the performance of
the model on the validation data. In essence, this tuning is a form of learning: a search
for a good configuration in some parameter space. As a result, tuning the configuration of the model based on its performance on the validation set can quickly result in
overfitting to the validation set, even though your model is never directly trained on it.

 Central to this phenomenon is the notion of information leaks. Every time you tune
a hyperparameter of your model based on the model’s performance on the validation
set, some information about the validation data leaks into the model. If you do this
only once, for one parameter, then very few bits of information will leak, and your validation set will remain reliable to evaluate the model. But if you repeat this many
times—running one experiment, evaluating on the validation set, and modifying your
model as a result—then you’ll leak an increasingly significant amount of information
about the validation set into the model.

At the end of the day, you’ll end up with a model that performs artificially well on
the validation data, because that’s what you optimized it for. You care about performance on completely new data, not the validation data, so you need to use a completely different, never-before-seen dataset to evaluate the model: the test dataset. Your
model shouldn’t have had access to any information about the test set, even indirectly

If anything about the model has been tuned based on test set performance, then your
measure of generalization will be flawed.

#### 4.2.2 Things to keep in mind

Keep an eye out for the following when you’re choosing an evaluation protocol:

Data representativeness—You want both your training set and test set to be representative of the data at hand. For instance, if you’re trying to classify images of
digits, and you’re starting from an array of samples where the samples are
ordered by their class, taking the first 80% of the array as your training set and
the remaining 20% as your test set will result in your training set containing
only classes 0–7, whereas your test set contains only classes 8–9. This seems like
a ridiculous mistake, but it’s surprisingly common. For this reason, you usually
should randomly shuffle your data before splitting it into training and test sets.

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it, because doing so will create a
temporal leak: your model will effectively be trained on data from the future. In
such situations, you should always make sure all data in your test set is posterior
to the data in the training set.

Redundancy in your data—If some data points in your data appear twice (fairly
common with real-world data), then shuffling the data and splitting it into a
training set and a validation set will result in redundancy between the training
and validation sets. In effect, you’ll be testing on part of your training data,
which is the worst thing you can do! Make sure your training set and validation
set are disjoint. 

### 4.3 Data preprocessing, feature engineering, and feature learning

In addition to model evaluation, an important question we must tackle before we dive
deeper into model development is the following: how do you prepare the input data
and targets before feeding them into a neural network? Many data-preprocessing and
feature-engineering techniques are domain specific

#### 4.4 Overfitting and underfitting

Overfitting happens in every
machine-learning problem. Learning how to deal with overfitting is essential to mastering machine learning

The fundamental issue in machine learning is the tension between optimization
and generalization. Optimization refers to the process of adjusting a model to get the
best performance possible on the training data (the learning in machine learning),
whereas generalization refers to how well the trained model performs on data it has
never seen before.

At the beginning of training, optimization and generalization are correlated: the
lower the loss on training data, the lower the loss on test data. While this is happening,
your model is said to be underfit: there is still progress to be made; the network hasn’t
yet modeled all relevant patterns in the training data. But after a certain number of
iterations on the training data, generalization stops improving, and validation metrics
stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning
to learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

To prevent a model from learning misleading or irrelevant patterns found in the
training data, the best solution is to get more training data. A model trained on more data
will naturally generalize better. When that isn’t possible, the next-best solution is to
modulate the quantity of information that your model is allowed to store or to add
constraints on what information it’s allowed to store. If a network can only afford to
memorize a small number of patterns, the optimization process will force it to focus
on the most prominent patterns, which have a better chance of generalizing well.

The processing of fighting overfitting this way is called regularization.

#### 4.4.1 Reducing the network’s size

The simplest way to prevent overfitting is to reduce the size of the model: the number
of learnable parameters in the model

In deep learning, the number of learnable parameters in a model is often referred to as the model’s capacity

 Intuitively, a model with
more parameters has more memorization capacity and therefore can easily learn a perfect dictionary-like mapping between training samples and their targets—a mapping
without any generalization power.

Always keep this in mind: deeplearning models tend to be good at fitting to the training data, but the real challenge
is generalization, not fitting.

On the other hand, if the network has limited memorization resources, it won’t be
able to learn this mapping as easily; thus, in order to minimize its loss, it will have to
resort to learning compressed representations that have predictive power regarding
the targets—precisely the type of representations we’re interested in. At the same
time, keep in mind that you should use models that have enough parameters that they
don’t underfit: your model shouldn’t be starved for memorization resources. There is
a compromise to be found between too much capacity and not enough capacity.

You must evaluate an array of different architectures (on your validation set, not on your test set, of course) in order to find the
correct model size for your data. The general workflow to find an appropriate model
size is to start with relatively few layers and parameters, and increase the size of the layers or add new layers until you see diminishing returns with regard to validation loss.

#### 4.4.2 Adding weight regularization

given some training data and a network architecture, multiple sets of weight
values (multiple models) could explain the data. Simpler models are less likely to overfit than complex ones.

A simple model in this context is a model where the distribution of parameter values
has less entropy (or a model with fewer parameters, as you saw in the previous section). Thus a common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights to take only small values, which makes the
distribution of weight values more regular. This is called weight regularization, and it’s
done by adding to the loss function of the network a cost associated with having large
weights.

 L1 regularization—The cost added is proportional to the absolute value of the
weight coefficients (the L1 norm of the weights).
 L2 regularization—The cost added is proportional to the square of the value of the
weight coefficients (the L2 norm of the weights). L2 regularization is also called
weight decay in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically the same as L2 regularization.

In Keras, weight regularization is added by passing weight regularizer instances to layers
as keyword arguments. Let’s add L2 weight regularization to the movie-review classification network.

#### 4.4.3 Adding dropout

Dropout is one of the most effective and most commonly used regularization techniques for neural networks,

Dropout, applied to a layer, consists of randomly dropping out
(setting to zero) a number of output features of the layer during training. Let’s say a
given layer would normally return a vector [0.2, 0.5, 1.3, 0.8, 1.1] for a given input
sample during training. After applying dropout, this vector will have a few zero entries
distributed at random: for example, [0, 0.5, 1.3, 0, 1.1].

 The dropout rate is the fraction
of the features that are zeroed out; it’s usually set between 0.2 and 0.5. At test time, no
units are dropped out; instead, the layer’s output values are scaled down by a factor
equal to the dropout rate, to balance for the fact that more units are active than at
training time.

randomly removing a different
subset of neurons on each example would prevent conspiracies and thus reduce overfitting

The core idea is that introducing noise in the output values of a layer can
break up happenstance patterns that aren’t significant (what Hinton refers to as conspiracies), which the network will start memorizing if no noise is present.

 In Keras, you can introduce dropout in a network via the Dropout layer, which is
applied to the output of the layer right before it:

### 4.5 The universal workflow of machine learning

In this section, we’ll present a universal blueprint that you can use to attack and solve
any machine-learning problem. The blueprint ties together the concepts you’ve
learned about in this chapter: problem definition, evaluation, feature engineering,
and fighting overfitting.

#### 4.5.1 Defining the problem and assembling a dataset

First, you must define the problem at hand:
 What will your input data be? What are you trying to predict? You can only learn
to predict something if you have available training data: for example, you can
only learn to classify the sentiment of movie reviews if you have both movie
reviews and sentiment annotations available. As such, data availability is usually
the limiting factor at this stage (unless you have the means to pay people to collect data for you).
 What type of problem are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Something else, like clustering, generation, or reinforcement learning?
Identifying the problem type will guide your choice of model architecture, loss
function, and so on

You can’t move to the next stage until you know what your inputs and outputs are, and
what data you’ll use. Be aware of the hypotheses you make at this stage:
 You hypothesize that your outputs can be predicted given your inputs.
 You hypothesize that your available data is sufficiently informative to learn the
relationship between inputs and outputs.

One class of unsolvable problems you should be aware of is nonstationary problems

One big issue is that the kinds of clothes people buy change from season
to season: clothes buying is a nonstationary phenomenon over the scale of a few
months. What you’re trying to model changes over time. 

In this case, the right move is
to constantly retrain your model on data from the recent past, or gather data at a
timescale where the problem is stationary. For a cyclical problem like clothes buying, a
few years’ worth of data will suffice to capture seasonal variation—but remember to
make the time of the year an input of your model!

Keep in mind that machine learning can only be used to memorize patterns that
are present in your training data. You can only recognize what you’ve seen before.
Using machine learning trained on past data to predict the future is making the
assumption that the future will behave like the past. That often isn’t the case. 

#### 4.5.2 Choosing a measure of success

To control something, you need to be able to observe it. To achieve success, you must
define what you mean by success—accuracy? Precision and recall? Customer-retention
rate? Your metric for success will guide the choice of a loss function: what your model
will optimize. It should directly align with your higher-level goals, such as the success
of your business

For balanced-classification problems, where every class is equally likely, accuracy and
area under the receiver operating characteristic curve (ROC AUC) are common metrics. For
class-imbalanced problems, you can use precision and recall. For ranking problems or
multilabel classification, you can use mean average precision. And it isn’t uncommon
to have to define your own custom metric by which to measure success. To get a sense
of the diversity of machine-learning success metrics and how they relate to different
problem domains, it’s helpful to browse the data science competitions on Kaggle
(https://kaggle.com); they showcase a wide range of problems and evaluation metrics.

#### 4.5.3 Deciding on an evaluation protocol

Once you know what you’re aiming for, you must establish how you’ll measure your
current progress. We’ve previously reviewed three common evaluation protocols:
 Maintaining a hold-out validation set—The way to go when you have plenty of
data
 Doing K-fold cross-validation—The right choice when you have too few samples
for hold-out validation to be reliable
 Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available
Just pick one of these. In most cases, the first will work well enough. 

#### 4.5.4 Preparing your data

Once you know what you’re training on, what you’re optimizing for, and how to evaluate your approach, you’re almost ready to begin training models. But first, you should
format your data in a way that can be fed into a machine-learning model—here, we’ll
assume a deep neural network:
 As you saw previously, your data should be formatted as tensors.
 The values taken by these tensors should usually be scaled to small values: for
example, in the [-1, 1] range or [0, 1] range.
 If different features take values in different ranges (heterogeneous data), then
the data should be normalized.
 You may want to do some feature engineering, especially for small-data problems.
Once your tensors of input data and target data are ready, you can begin to train models. 

#### 4.5.5 Developing a model that does better than a baseline

Your goal at this stage is to achieve statistical power: that is, to develop a small model
that is capable of beating a dumb baseline. In the MNIST digit-classification example,
anything that achieves an accuracy greater than 0.1 can be said to have statistical
power; in the IMDB example, it’s anything with an accuracy greater than 0.5.

Note that it’s not always possible to achieve statistical power. If you can’t beat a random baseline after trying multiple reasonable architectures, it may be that the answer
to the question you’re asking isn’t present in the input data. Remember that you make
two hypotheses:

 You hypothesize that your outputs can be predicted given your inputs.
 You hypothesize that the available data is sufficiently informative to learn the
relationship between inputs and outputs.

Assuming that things go well, you need to make three key choices to build your
first working model:
 Last-layer activation—This establishes useful constraints on the network’s output. For instance, the IMDB classification example used sigmoid in the last
layer; the regression example didn’t use any last-layer activation; and so on.
 Loss function—This should match the type of problem you’re trying to solve. For
instance, the IMDB example used binary_crossentropy, the regression example used mse, and so on.
 Optimization configuration—What optimizer will you use? What will its learning
rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.

Regarding the choice of a loss function, note that it isn’t always possible to directly
optimize for the metric that measures success on a problem. Sometimes there is no
easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-batch of data (ideally, a loss function should be computable
for as little as a single data point) and must be differentiable (otherwise, you can’t use
backpropagation to train your network).

In general, you
can hope that the lower the crossentropy gets, the higher the ROC AUC will be.

Problem type: Last-layer activation: Loss function
Binary classification: sigmoid: binary_crossentropy
Multiclass, single-label classification: softmax: categorical_crossentropy
Multiclass, multilabel classification: sigmoid: binary_crossentropy
Regression to arbitrary values: None: mse
Regression to values between 0 and 1: sigmoid: mse or binary_crossentropy

#### 4.5.6 Scaling up: developing a model that overfits

Once you’ve obtained a model that has statistical power, the question becomes, is your
model sufficiently powerful? Does it have enough layers and parameters to properly
model the problem at hand? For instance, a network with a single hidden layer with
two units would have statistical power on MNIST but wouldn’t be sufficient to solve the
problem well. Remember that the universal tension in machine learning is between
optimization and generalization; the ideal model is one that stands right at the border
between underfitting and overfitting; between undercapacity and overcapacity. To figure out where this border lies, first you must cross it

To figure out how big a model you’ll need, you must develop a model that overfits.
This is fairly easy:
1 Add layers.
2 Make the layers bigger.
3 Train for more epochs.

Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.
 The next stage is to start regularizing and tuning the model, to get as close as possible to the ideal model that neither underfits nor overfits. 

#### 4.5.7 Regularizing your model and tuning your hyperparameters

This step will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and
repeat, until the model is as good as it can get.

These are some things you should try:
 Add dropout.
 Try different architectures: add or remove layers.
 Add L1 and/or L2 regularization.
 Try different hyperparameters (such as the number of units per layer or the
learning rate of the optimizer) to find the optimal configuration.
 Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

Be mindful of the following: every time you use feedback from your validation process
to tune your model, you leak information about the validation process into the model.
Repeated just a few times, this is innocuous; but done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even
though no model is directly trained on any of the validation data). This makes the
evaluation process less reliable.

 Once you’ve developed a satisfactory model configuration, you can train your final
production model on all the available data (training and validation) and evaluate it
one last time on the test set. If it turns out that performance on the test set is significantly worse than the performance measured on the validation data, this may mean
either that your validation procedure wasn’t reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case,
you may want to switch to a more reliable evaluation protocol (such as iterated K-fold
validation). 

### Part 2 Deep Learning in Practice

Chapters 5–9 will help you gain practical intuition about how to solve realworld problems using deep learning, and will familiarize you with essential deeplearning best practices. Most of the code examples in the book are concentrated
in this second half.

### Chapter 5 Deep learning for computer vision

This chapter introduces convolutional neural networks, also known as convnets, a
type of deep-learning model almost universally used in computer vision applications. You’ll learn to apply convnets to image-classification problems—in particular
those involving small training datasets, which are the most common use case if you
aren’t a large tech company.

#### 5.1 Introduction to convnets

Even though
the convnet will be basic, its accuracy will blow out of the water that of the densely
connected model from chapter 2.

The following lines of code show you what a basic convnet looks like. It’s a stack of
Conv2D and MaxPooling2D layers

Importantly, a convnet takes as input tensors of shape (image_height, image_width,
image_channels) (not including the batch dimension). In this case, we’ll configure
the convnet to process inputs of size (28, 28, 1), which is the format of MNIST
images. We’ll do this by passing the argument input_shape=(28, 28, 1) to the first
layer.

You can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of
shape (height, width, channels).

 The width and height dimensions tend to shrink as you go deeper in the network. The number of channels is controlled by the first
argument passed to the Conv2D layers (32 or 64).

Whereas the densely connected network from chapter 2 had a test accuracy of 97.8%,
the basic convnet has a test accuracy of 99.3%: we decreased the error rate by 68%
(relative). Not bad!
 But why does this simple convnet work so well, compared to a densely connected
model? To answer this, let’s dive into what the Conv2D and MaxPooling2D layers do.

#### 5.1.1 The convolution operation

The fundamental difference between a densely connected layer and a convolution
layer is this: Dense layers learn global patterns in their input feature space (for example, for a MNIST digit, patterns involving all pixels), whereas convolution layers learn
local patterns (see figure 5.1): in the case of images, patterns found in small 2D windows of the inputs.

This key characteristic gives convnets two interesting properties:

 The patterns they learn are translation invariant. After learning a certain pattern in
the lower-right corner of a picture, a convnet can recognize it anywhere: for
example, in the upper-left corner. A densely connected network would have to
learn the pattern anew if it appeared at a new location. This makes convnets
data efficient when processing images (because the visual world is fundamentally
translation invariant): they need fewer training samples to learn representations
that have generalization power.

 They can learn spatial hierarchies of patterns (see figure 5.2). A first convolution layer
will learn small local patterns such as edges, a second convolution layer will
learn larger patterns made of the features of the first layers, and so on. This
allows convnets to efficiently learn increasingly complex and abstract visual concepts 

Convolutions operate over 3D tensors, called feature maps, with two spatial axes (height
and width) as well as a depth axis (also called the channels axis). For an RGB image, the
dimension of the depth axis is 3, because the image has three color channels: red,
green, and blue. For a black-and-white picture, like the MNIST digits, the depth is 1
(levels of gray). The convolution operation extracts patches from its input feature
map and applies the same transformation to all of these patches, producing an output
feature map

This output feature map is still a 3D tensor: it has a width and a height. Its
depth can be arbitrary, because the output depth is a parameter of the layer, and the
different channels in that depth axis no longer stand for specific colors as in RGB
input; rather, they stand for filters. Filters encode specific aspects of the input data: at a
high level, a single filter could encode the concept “presence of a face in the input,”
for instance.

In Keras Conv2D layers, these parameters are the first arguments passed to the layer:
Conv2D(output_depth, (window_height, window_width)).

#### 5.1.2 The max-pooling operation

you may have noticed that the size of the feature maps is
halved after every MaxPooling2D layer. For instance, before the first MaxPooling2D layers, the feature map is 26 × 26, but the max-pooling operation halves it to 13 × 13.
That’s the role of max pooling: to aggressively downsample feature maps, much like
strided convolutions.

Max pooling consists of extracting windows from the input feature maps and outputting the max value of each channel. It’s conceptually similar to convolution, except
that instead of transforming local patches via a learned linear transformation (the convolution kernel), they’re transformed via a hardcoded max tensor operation.

What’s wrong with this setup? Two things:
 It isn’t conducive to learning a spatial hierarchy of features. The 3 × 3 windows
in the third layer will only contain information coming from 7 × 7 windows in
the initial input. The high-level patterns learned by the convnet will still be very
small with regard to the initial input, which may not be enough to learn to classify digits (try recognizing a digit by only looking at it through windows that are
7 × 7 pixels!). We need the features from the last convolution layer to contain
information about the totality of the input.
 The final feature map has 22 × 22 × 64 = 30,976 total coefficients per sample.
This is huge. If you were to flatten it to stick a Dense layer of size 512 on top,
that layer would have 15.8 million parameters. This is far too large for such a
small model and would result in intense overfitting.

In short, the reason to use downsampling is to reduce the number of feature-map
coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows 

the reason is that features tend to encode the spatial presence of some pattern
or concept over the different tiles of the feature map (hence, the term feature map),
and it’s more informative to look at the maximal presence of different features than at
their average presence. So the most reasonable subsampling strategy is to first produce
dense maps of features (via unstrided convolutions) and then look at the maximal
activation of the features over small patches, rather than looking at sparser windows of
the inputs (via strided convolutions) or averaging input patches, which could cause
you to miss or dilute feature-presence information.

#### 5.2 Training a convnet from scratch on a small dataset

Having to train an image-classification model using very little data is a common situation, which you’ll likely encounter in practice if you ever do computer vision in a
professional context. A “few” samples can mean anywhere from a few hundred to a
few tens of thousands of images.

training a new
model from scratch using what little data you have. You’ll start by naively training a
small convnet on the 2,000 training samples, without any regularization, to set a baseline for what can be achieved. This will get you to a classification accuracy of 71%.

At
that point, the main issue will be overfitting. Then we’ll introduce data augmentation, a
powerful technique for mitigating overfitting in computer vision. By using data augmentation, you’ll improve the network to reach an accuracy of 82%.

In the next section, we’ll review two more essential techniques for applying deep
learning to small datasets: feature extraction with a pretrained network (which will get you
to an accuracy of 90% to 96%) and fine-tuning a pretrained network (this will get you to a
final accuracy of 97%)

 Together, these three strategies—training a small model from
scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

#### 5.2.1 The relevance of deep learning for small-data problems

You’ll sometimes hear that deep learning only works when lots of data is available.
This is valid in part: one fundamental characteristic of deep learning is that it can find
interesting features in the training data on its own, without any need for manual feature engineering, and this can only be achieved when lots of training examples are
available. This is especially true for problems where the input samples are very highdimensional, like images.

But what constitutes lots of samples is relative—relative to the size and depth of the
network you’re trying to train, for starters. It isn’t possible to train a convnet to solve a
complex problem with just a few tens of samples, but a few hundred can potentially
suffice if the model is small and well regularized and the task is simple. Because convnets learn local, translation-invariant features, they’re highly data efficient on perceptual problems.

Training a convnet from scratch on a very small image dataset will still
yield reasonable results despite a relative lack of data, without the need for any custom
feature engineering. 

What’s more, deep-learning models are by nature highly repurposable: you can
take, say, an image-classification or speech-to-text model trained on a large-scale dataset
and reuse it on a significantly different problem with only minor changes

in the case of computer vision, many pretrained models (usually trained on the ImageNet dataset) are now publicly available for download and can be used to bootstrap powerful vision models out of very little data

#### 5.2.2  Downloading the data

 The pictures are medium-resolution color JPEGs.
 
Unsurprisingly, the dogs-versus-cats Kaggle competition in 2013 was won by entrants
who used convnets.

This dataset contains 25,000 images of dogs and cats (12,500 from each class) and
is 543 MB (compressed).

So you do indeed have 2,000 training images, 1,000 validation images, and 1,000 test
images. Each split contains the same number of samples from each class: this is a balanced binary-classification problem, which means classification accuracy will be an
appropriate measure of success. 

#### 5.2.3 Building your network

You built a small convnet for MNIST in the previous example, so you should be familiar with such convnets. You’ll reuse the same general structure: the convnet will be a
stack of alternated Conv2D (with relu activation) and MaxPooling2D layers.

But because you’re dealing with bigger images and a more complex problem, you’ll
make your network larger, accordingly: it will have one more Conv2D + MaxPooling2D
stage. This serves both to augment the capacity of the network and to further reduce
the size of the feature maps so they aren’t overly large when you reach the Flatten
layer.

Because you’re attacking a binary-classification problem, you’ll end the network with a
single unit (a Dense layer of size 1) and a sigmoid activation. This unit will encode the
probability that the network is looking at one class or the other.

For the compilation step, you’ll go with the RMSprop optimizer, as usual. Because you
ended the network with a single sigmoid unit, you’ll use binary crossentropy as the
loss

#### 5.2.4 Data preprocessing

As you know by now, data should be formatted into appropriately preprocessed floatingpoint tensors before being fed into the network. Currently, the data sits on a drive as
JPEG files, so the steps for getting it into the network are roughly as follows:

1 Read the picture files.
2 Decode the JPEG content to RGB grids of pixels.
3 Convert these into floating-point tensors.
4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know,
neural networks prefer to deal with small input values).

It may seem a bit daunting, but fortunately Keras has utilities to take care of these
steps automatically. Keras has a module with image-processing helper tools, located at
keras.preprocessing.image. In particular, it contains the class ImageDataGenerator,
which lets you quickly set up Python generators that can automatically turn image files
on disk into batches of preprocessed tensors.

Let’s look at the output of one of these generators: it yields batches of 150 × 150 RGB
images (shape (20, 150, 150, 3)) and binary labels (shape (20,)). There are 20 samples in each batch (the batch size).

Note that the generator yields these batches indefinitely: it loops endlessly over the images in the target folder. For this reason, you need
to break the iteration loop at some point:

Let’s fit the model to the data using the generator. You do so using the fit_generator
method, the equivalent of fit for data generators like this one. It expects as its first
argument a Python generator that will yield batches of inputs and targets indefinitely

Because the data is being generated endlessly, the Keras model
needs to know how many samples to draw from the generator before declaring an
epoch over. This is the role of the steps_per_epoch argument: after having drawn
steps_per_epoch batches from the generator

When using fit_generator, you can pass a validation_data argument, much as
with the fit method. It’s important to note that this argument is allowed to be a data
generator, but it could also be a tuple of Numpy arrays. If you pass a generator as
validation_data, then this generator is expected to yield batches of validation data
endlessly

These plots are characteristic of overfitting. The training accuracy increases linearly
over time, until it reaches nearly 100%, whereas the validation accuracy stalls at 70–72%.
The validation loss reaches its minimum after only five epochs and then stalls, whereas
the training loss keeps decreasing linearly until it reaches nearly 0.

Because you have relatively few training samples (2,000), overfitting will be your
number-one concern. You already know about a number of techniques that can help
mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now
going to work with a new one, specific to computer vision and used almost universally
when processing images with deep-learning models: data augmentation. 

#### 5.2.5 Using data augmentation

Overfitting is caused by having too few samples to learn from, rendering you unable
to train a model that can generalize to new data.

Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would
never overfit. Data augmentation takes the approach of generating more training data
from existing training samples, by augmenting the samples via a number of random
transformations that yield believable-looking images. The goal is that at training time,
your model will never see the exact same picture twice. This helps expose the model
to more aspects of the data and generalize better.

In Keras, this can be done by configuring a number of random transformations to
be performed on the images read by the ImageDataGenerator instance.

These are just a few of the options available (for more, see the Keras documentation).
Let’s quickly go over this code:
 rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
 width_shift and height_shift are ranges (as a fraction of total width or
height) within which to randomly translate pictures vertically or horizontally.
 shear_range is for randomly applying shearing transformations.
 zoom_range is for randomly zooming inside pictures.
 horizontal_flip is for randomly flipping half the images horizontally—relevant when there are no assumptions of horizontal asymmetry (for example,
real-world pictures).
 fill_mode is the strategy used for filling in newly created pixels, which can
appear after a rotation or a width/height shift.

If you train a new network using this data-augmentation configuration, the network
will never see the same input twice. But the inputs it sees are still heavily intercorrelated, because they come from a small number of original images—you can’t produce new information, you can only remix existing information. As such, this may not
be enough to completely get rid of overfitting.

. To further fight overfitting, you’ll also
add a Dropout layer to your model, right before the densely connected classifier.

 Thanks to data augmentation and dropout, you’re no longer overfitting: the training curves are closely tracking
the validation curves. You now reach an accuracy of 82%, a 15% relative improvement
over the non-regularized model.

By using regularization techniques even further, and by tuning the network’s parameters (such as the number of filters per convolution layer, or the number of layers in
the network), you may be able to get an even better accuracy, likely up to 86% or 87%.
But it would prove difficult to go any higher just by training your own convnet from
scratch, because you have so little data to work with. As a next step to improve your
accuracy on this problem, you’ll have to use a pretrained model, which is the focus of
the next two sections. 

#### 5.3 Using a pretrained convnet

A common and highly effective approach to deep learning on small image datasets is
to use a pretrained network. A pretrained network is a saved network that was previously
trained on a large dataset, typically on a large-scale image-classification task.

If this
original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained network can effectively act as a generic model of the
visual world, and hence its features can prove useful for many different computervision problems, even though these new problems may involve completely different
classes than those of the original task.

 Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow-learning approaches, and it
makes deep learning very effective for small-data problems.

There are two ways to use a pretrained network: feature extraction and fine-tuning.

#### 5.3.1 Feature Extraction

Feature extraction consists of using the representations learned by a previous network
to extract interesting features from new samples. These features are then run through
a new classifier, which is trained from scratch.

, convnets used for image classification comprise two parts:
they start with a series of pooling and convolution layers, and they end with a densely
connected classifier. The first part is called the convolutional base of the model.

In the
case of convnets, feature extraction consists of taking the convolutional base of a
previously trained network, running the new data through it, and training a new classifier on top of the output

 Could you reuse the densely connected classifier as well? In general, doing so should be avoided. The reason is that the representations learned by the convolutional base are likely to be more generic and therefore
more reusable:

the feature maps of a convnet are presence maps of generic concepts
over a picture, which is likely to be useful regardless of the computer-vision problem at
hand. But the representations learned by the classifier will necessarily be specific to the
set of classes on which the model was trained—they will only contain information about
the presence probability of this or that class in the entire picture.

Additionally, representations found in densely connected layers no longer contain any information about
where objects are located in the input image: these layers get rid of the notion of space,
whereas the object location is still described by convolutional feature maps. For problems where object location matters, densely connected features are largely useless.

Note that the level of generality (and therefore reusability) of the representations
extracted by specific convolution layers depends on the depth of the layer in the
model. Layers that come earlier in the model extract local, highly generic feature
maps (such as visual edges, colors, and textures), whereas layers that are higher up
extract more-abstract concepts (such as “cat ear” or “dog eye”).

So if your new dataset
differs a lot from the dataset on which the original model was trained, you may be better off using only the first few layers of the model to do feature extraction, rather than
using the entire convolutional base.

#### 5.3.2 Fine-tuning

Another widely used technique for model reuse, complementary to feature
extraction, is fine-tuning

 Fine-tuning consists of unfreezing a few of
the top layers of a frozen model base used for feature extraction, and jointly training
both the newly added part of the model (in this case, the fully connected classifier)
and these top layers.

This is called fine-tuning because it slightly adjusts the more
abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

the
steps for fine-tuning a network are as follow:
1 Add your custom network on top of an already-trained base network.
2 Freeze the base network.
3 Train the part you added.
4 Unfreeze some layers in the base network.
5 Jointly train both these layers and the part you added.

#### 5.3.3 Wrapping up

 Convnets are the best type of machine-learning models for computer-vision
tasks. It’s possible to train one from scratch even on a very small dataset, with
decent results.
 On a small dataset, overfitting will be the main issue. Data augmentation is a
powerful way to fight overfitting when you’re working with image data.
 It’s easy to reuse an existing convnet on a new dataset via feature extraction.
This is a valuable technique for working with small image datasets.
 As a complement to feature extraction, you can use fine-tuning, which adapts to
a new problem some of the representations previously learned by an existing
model. This pushes performance a bit further.

### 5.4 Visualizing what convnets learn

The representations learned by convnets are highly amenable to visualization, in large part because they’re representations of visual concepts

 Visualizing intermediate convnet outputs (intermediate activations)—Useful for
understanding how successive convnet layers transform their input, and for getting a first idea of the meaning of individual convnet filters.
 Visualizing convnets filters—Useful for understanding precisely what visual pattern or concept each filter in a convnet is receptive to.
 Visualizing heatmaps of class activation in an image—Useful for understanding
which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images.

#### 5.4.1 Visualizing intermediate activations

Visualizing intermediate activations consists of displaying the feature maps that are
output by various convolution and pooling layers in a network, given a certain input

This gives a view into how an input is decomposed into the different filters
learned by the network. You want to visualize feature maps with three dimensions:
width, height, and depth (channels). 

Each channel encodes relatively independent
features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image

In order to extract the feature maps you want to look at, you’ll create a Keras model
that takes batches of images as input, and outputs the activations of all convolution and
pooling layers

When fed an image input, this model returns the values of the layer activations in the
original model

There are a few things to note here:
 The first layer acts as a collection of various edge detectors. At that stage, the
activations retain almost all of the information present in the initial picture.
 As you go higher, the activations become increasingly abstract and less visually
interpretable. They begin to encode higher-level concepts such as “cat ear” and
“cat eye.” Higher presentations carry increasingly less information about the
visual contents of the image, and increasingly more information related to the
class of the image.
 The sparsity of the activations increases with the depth of the layer: in the first
layer, all filters are activated by the input image; but in the following layers,
more and more filters are blank. This means the pattern encoded by the filter
isn’t found in the input image.

We have just evidenced an important universal characteristic of the representations
learned by deep neural networks: the features extracted by a layer become increasingly abstract with the depth of the layer. The activations of higher layers carry less
and less information about the specific input being seen, and more and more information about the target

 This is analogous to the way humans and animals perceive the world: after observing a scene for a few seconds, a human can remember which abstract objects were
present in it (bicycle, tree) but can’t remember the specific appearance of these
objects

#### 5.4.2 Visualizing convnet filters

Another easy way to inspect the filters learned by convnets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in
input space: applying gradient descent to the value of the input image of a convnet so as
to maximize the response of a specific filter, starting from a blank input image. The
resulting input image will be one that the chosen filter is maximally responsive to.

The process is simple: you’ll build a loss function that maximizes the value of a
given filter in a given convolution layer, and then you’ll use stochastic gradient
descent to adjust the values of the input image so as to maximize this activation value.

To implement gradient descent, you’ll need the gradient of this loss with respect to
the model’s input. To do this, you’ll use the gradients function packaged with the
backend module of Keras

A non-obvious trick to use to help the gradient-descent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the average
of the square of the values in the tensor). This ensures that the magnitude of the
updates done to the input image is always within the same range.

 Let’s put them together into a Python function that takes
as input a layer name and a filter index, and returns a valid image tensor representing
the pattern that maximizes the activation of the specified filter.

These filter visualizations tell you a lot about how convnet layers see the world: each
layer in a convnet learns a collection of filters such that their inputs can be expressed
as a combination of the filters.

The filters in these convnet filter banks
get increasingly complex and refined as you go higher in the model:
 The filters from the first layer in the model (block1_conv1) encode simple
directional edges and colors (or colored edges, in some cases).
 The filters from block2_conv1 encode simple textures made from combinations of edges and colors.
 The filters in higher layers begin to resemble textures found in natural images:
feathers, eyes, leaves, and so on. 

#### 5.4.3 Visualizing heatmaps of class activation

 It also allows you to locate specific objects in an image
 
This general category of techniques is called class activation map (CAM) visualization,
and it consists of producing heatmaps of class activation over input images. 

A class activation heatmap is a 2D grid of scores associated with a specific output class, computed
for every location in any input image, indicating how important each location is with respect to the class under consideration

Intuitively, one way to understand this trick is that you’re
weighting a spatial map of “how intensely the input image activates different channels” by “how important each channel is with regard to the class,” resulting in a spatial
map of “how intensely the input image activates the class.”

Chapter summary
 Convnets are the best tool for attacking visual-classification problems.
 Convnets work by learning a hierarchy of modular patterns and concepts
to represent the visual world.
 The representations they learn are easy to inspect—convnets are the
opposite of black boxes!
 You’re now capable of training your own convnet from scratch to solve an
image-classification problem.
 You understand how to use visual data augmentation to fight overfitting.
 You know how to use a pretrained convnet to do feature extraction and
fine-tuning.
 You can generate visualizations of the filters learned by your convnets, as
well as heatmaps of class activity

### 6.1 Working with Text Data 

The deep-learning sequence-processing models introduced in
the following sections can use text to produce a basic form of natural-language understanding, sufficient for applications including document classification, sentiment
analysis, author identification, and even question-answering (QA) (in a constrained
context).

none of these deeplearning models truly understand text in a human sense; rather, these models can
map the statistical structure of written language, which is sufficient to solve many simple textual tasks.

Deep learning for natural-language processing is pattern recognition
applied to words, sentences, and paragraphs, in much the same way that computer
vision is pattern recognition applied to pixels

 Like all other neural networks, deep-learning models don’t take as input raw text:
they only work with numeric tensors. Vectorizing text is the process of transforming text
into numeric tensors. This can be done in multiple ways:
 Segment text into words, and transform each word into a vector.
 Segment text into characters, and transform each character into a vector.
 Extract n-grams of words or characters, and transform each n-gram into a vector.
N-grams are overlapping groups of multiple consecutive words or characters.

Collectively, the different units into which you can break down text (words, characters, or n-grams) are called tokens, and breaking text into such tokens is called tokenization. All text-vectorization processes consist of applying some tokenization scheme and
then associating numeric vectors with the generated tokens. 

These vectors, packed
into sequence tensors, are fed into deep neural networks. There are multiple ways to
associate a vector with a token. In this section, I’ll present two major ones: one-hot
encoding of tokens, and token embedding (typically used exclusively for words, and called
word embedding). 

#### 6.1.1 One-hot encoding of words and characters

One-hot encoding is the most common, most basic way to turn a token into a vector.

. It consists of associating a unique integer index with every word
and then turning this integer index i into a binary vector of size N (the size of the
vocabulary); the vector is all zeros except for the ith entry, which is 1.

 Of course, one-hot encoding can be done at the character level, as well
 
A variant of one-hot encoding is the so-called one-hot hashing trick, which you can use
when the number of unique tokens in your vocabulary is too large to handle explicitly.
Instead of explicitly assigning an index to each word and keeping a reference of these
indices in a dictionary, you can hash words into vectors of fixed size.

This is typically
done with a very lightweight hashing function. The main advantage of this method is
that it does away with maintaining an explicit word index, which saves memory and
allows online encoding of the data

The one drawback of this approach is that it’s
susceptible to hash collisions: two different words may end up with the same hash, and
subsequently any machine-learning model looking at these hashes won’t be able to tell
the difference between these words.

The likelihood of hash collisions decreases when
the dimensionality of the hashing space is much larger than the total number of
unique tokens being hashed.

#### 6.1.2 Using word embeddings

Another popular and powerful way to associate a vector with a word is the use of dense
word vectors, also called word embeddings

Whereas the vectors obtained through one-hot
encoding are binary, sparse (mostly made of zeros), and very high-dimensional (same
dimensionality as the number of words in the vocabulary), word embeddings are lowdimensional floating-point vectors

Unlike the word vectors obtained via one-hot encoding, word
embeddings are learned from data. It’s common to see word embeddings that are
256-dimensional, 512-dimensional, or 1,024-dimensional when dealing with very large
vocabularies. On the other hand, one-hot encoding words generally leads to vectors
that are 20,000-dimensional or greater

So, word embeddings pack more information into far fewer dimensions.

There are two ways to obtain word embeddings:
 Learn word embeddings jointly with the main task you care about (such as document classification or sentiment prediction). In this setup, you start with random word vectors and then learn word vectors in the same way you learn the
weights of a neural network.
 Load into your model word embeddings that were precomputed using a different machine-learning task than the one you’re trying to solve. These are called
pretrained word embeddings.

The simplest way to associate a dense vector with a word is to choose the vector at
random. The problem with this approach is that the resulting embedding space has
no structure: for instance, the words accurate and exact may end up with completely
different embeddings, even though they’re interchangeable in most sentences. It’s
difficult for a deep neural network to make sense of such a noisy, unstructured
embedding space.

To get a bit more abstract, the geometric relationships between word vectors
should reflect the semantic relationships between these words. Word embeddings are
meant to map human language into a geometric space. For instance, in a reasonable
embedding space, you would expect synonyms to be embedded into similar word vectors; and in general, you would expect the geometric distance (such as L2 distance)
between any two word vectors to relate to the semantic distance between the associated words

. In addition to distance, you may want
specific directions in the embedding space to be meaningful.

the perfect word-embedding space for an English-language movie-review sentimentanalysis model may look different from the perfect embedding space for an Englishlanguage legal-document-classification model, because the importance of certain
semantic relationships varies from task to task.

Sometimes, you have so little training data available that you can’t use your data
alone to learn an appropriate task-specific embedding of your vocabulary. What do
you do then?
 Instead of learning word embeddings jointly with the problem you want to solve,
you can load embedding vectors from a precomputed embedding space that you
know is highly structured and exhibits useful properties—that captures generic
aspects of language structure. The rationale behind using pretrained word embeddings in natural-language processing is much the same as for using pretrained convnets in image classification: you don’t have enough data available to learn truly
powerful features on your own, but you expect the features that you need to be fairly
generic—that is, common visual features or semantic features. In this case, it makes
sense to reuse features learned on a different problem

#### 6.1.3 Putting it all together: from raw text to word embeddings

You’ll use a model similar to the one we just went over: embedding sentences in
sequences of vectors, flattening them, and training a Dense layer on top. But you’ll do
so using pretrained word embeddings; and instead of using the pretokenized IMDB
data packaged in Keras, you’ll start from scratch by downloading the original text data.

#### 6.1.4 Wrapping up

Now you’re able to do the following:
 Turn raw text into something a neural network can process
 Use the Embedding layer in a Keras model to learn task-specific token embeddings
 Use pretrained word embeddings to get an extra boost on small naturallanguage-processing problems 

### 6.2 Understanding recurrent neural networks

A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and convnets, is that they have no memory. Each input shown to
them is processed independently, with no state kept in between inputs. With such networks, in order to process a sequence or a temporal series of data points, you have to
show the entire sequence to the network at once: turn it into a single data point. 
feedforward networks.

A recurrent neural network (RNN) adopts the same principle, albeit in an extremely
simplified version: it processes sequences by iterating through the sequence elements
and maintaining a state containing information relative
to what it has seen so far. 

 In effect, an RNN is a type of
neural network that has an internal loop (see figure 6.9).
The state of the RNN is reset between processing two different, independent sequences (such as two different
IMDB reviews), so you still consider one sequence a single data point: a single input to the network. What
changes is that this data point is no longer processed in a
single step; rather, the network internally loops over
sequence elements.

#### 6.2.1 A recurrent layer in Keras

The process you just naively implemented in Numpy corresponds to an actual Keras
layer—the SimpleRNN layer:
from keras.layers import SimpleRNN
There is one minor difference: SimpleRNN processes batches of sequences, like all other
Keras layers, not a single sequence as in the Numpy example. This means it takes inputs
of shape

 Like all recurrent layers in Keras, SimpleRNN can be run in two different modes: it
can return either the full sequences of successive outputs for each timestep (a 3D tensor of shape (batch_size, timesteps, output_features)) or only the last output for
each input sequence (a 2D tensor of shape (batch_size, output_features)). These
two modes are controlled by the return_sequences constructor argument. 

#### 6.2.2 Understanding the LSTM and GRU layers

SimpleRNN isn’t the only recurrent layer available in Keras. There are two others: LSTM
and GRU. . In practice, you’ll always use one of these, because SimpleRNN is generally too
simplistic to be of real use.

. SimpleRNN has a major issue: although it should theoretically
be able to retain at time t information about inputs seen many timesteps before, in
practice, such long-term dependencies are impossible to learn. This is due to the vanishing gradient problem, an effect that is similar to what is observed with non-recurrent
networks (feedforward networks) that are many layers deep: as you keep adding layers
to a network, the network eventually becomes untrainable.

LSTM layer is a variant of the SimpleRNN layer you already know about; it adds a way
to carry information across many timesteps. Imagine a conveyor belt running parallel
to the sequence you’re processing. Information from the sequence can jump onto the
conveyor belt at any point, be transported to a later timestep, and jump off, intact,
when you need it. This is essentially what LSTM does: it saves information for later,
thus preventing older signals from gradually vanishing during processing.

#### 6.2.3 A concrete LSTM example in Keras

 you’ll set up a model using an LSTM layer
and train it on the IMDB data

Why isn’t LSTM performing better? One reason is that you made no effort
to tune hyperparameters such as the embeddings dimensionality or the LSTM output
dimensionality. Another may be lack of regularization. But honestly, the primary reason is that analyzing the global, long-term structure of the reviews (what LSTM is good
at) isn’t helpful for a sentiment-analysis problem. Such a basic problem is well solved
by looking at what words occur in each review, and at what frequency. That’s what the
first fully connected approach looked at. But there are far more difficult naturallanguage-processing problems out there, where the strength of LSTM will become
apparent: in particular, question-answering and machine translation.

#### 6.2.4 Wrapping up

Now you understand the following:
 What RNNs are and how they work
 What LSTM is, and why it works better on long sequences than a naive RNN
 How to use Keras RNN layers to process sequence data

### 6.3 Advanced use of recurrent neural networks

In this section, we’ll review three advanced techniques for improving the performance and generalization power of recurrent neural networks. By the end of the section, you’ll know most of what there is to know about using recurrent networks with
Keras. We’ll demonstrate all three concepts on a temperature-forecasting problem,
where you have access to a timeseries of data points coming from sensors installed on
the roof of a building, such as temperature, air pressure, and humidity, which you use
to predict what the temperature will be 24 hours after the last data point.

 Recurrent dropout—This is a specific, built-in way to use dropout to fight overfitting in recurrent layers.
 Stacking recurrent layers—This increases the representational power of the network (at the cost of higher computational loads).
 Bidirectional recurrent layers—These present the same information to a recurrent

#### 6.3.1 A temperature-forecasting problem

 In this dataset, 14 different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but this example is limited to data
from 2009–2016. This dataset is perfect for learning to work with numerical
timeseries. You’ll use it to build a model that takes as input some data from the recent
past (a few days’ worth of data points) and predicts the air temperature 24 hours in
the future.

 If you were trying to predict average temperature for the next month given a few
months of past data, the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the data over a scale of days, the temperature looks a
lot more chaotic. Is this timeseries predictable at a daily scale? Let’s find out.

#### 6.3.2 Preparing the data

 given data going as far back
as lookback timesteps (a timestep is 10 minutes) and sampled every steps timesteps,
can you predict the temperature in delay timesteps? You’ll use the following parameter values:
 lookback = 720—Observations will go back 5 days.
 steps = 6—Observations will be sampled at one data point per hour.
 delay = 144—Targets will be 24 hours in the fu

To get started, you need to do two things:
 Preprocess the data to a format a neural network can ingest. This is easy: the
data is already numerical, so you don’t need to do any vectorization. But each
timeseries in the data is on a different scale (for example, temperature is typically between -20 and +30, but atmospheric pressure, measured in mbar, is
around 1,000). You’ll normalize each timeseries independently so that they all
take small values on a similar scale.
 Write a Python generator that takes the current array of float data and yields
batches of data from the recent past, along with a target temperature in the
future. Because the samples in the dataset are highly redundant (sample N and
sample N + 1 will have most of their timesteps in common), it would be wasteful
to explicitly allocate every samp

You’ll preprocess the data by subtracting the mean of each timeseries and dividing by
the standard deviation. You’re going to use the first 200,000 timesteps as training data,
so compute the mean and standard deviation only on this fraction of the data.

Listing 6.33 shows the data generator you’ll use. It yields a tuple (samples, targets),
where samples is one batch of input data and targets is the corresponding array of
target temperatures. It takes the following arguments

 data—The original array of floating-point data, which you normalized in listing 6.32.
 lookback—How many timesteps back the input data should go.
 delay—How many timesteps in the future the target should be.
 min_index and max_index—Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
 shuffle—Whether to shuffle the samples or draw them in chronological order.
 batch_size—The number of samples per batch.
 step—The period, in timesteps, at which you sample data. You’ll set it to 6 in
order to draw one data point every hour.

#### 6.3.3 A common-sense, non-machine-learning baseline

Before you start using black-box deep-learning models to solve the temperatureprediction problem, let’s try a simple, common-sense approach. It will serve as a sanity
check, and it will establish a baseline that you’ll have to beat in order to demonstrate
the usefulness of more-advanced machine-learning models. Such common-sense baselines can be useful when you’re approaching a new problem for which there is no
known solution (yet).

A classic example is that of unbalanced classification tasks,
where some classes are much more common than others. If your dataset contains 90%
instances of class A and 10% instances of class B, then a common-sense approach to
the classification task is to always predict “A” when presented with a new sample. Such
a classifier is 90% accurate overall, and any learning-based approach should therefore
beat this 90% score in order to demonstrate usefulness. Sometimes, such elementary
baselines can prove surprisingly hard to beat.

 In this case, the temperature timeseries can safely be assumed to be continuous
(the temperatures tomorrow are likely to be close to the temperatures today) as well
as periodical with a daily period. Thus a common-sense approach is to always predict
that the temperature 24 hours from now will be equal to the temperature right now

#### 6.3.4 A basic machine-learning approach

In the same way that it’s useful to establish a common-sense baseline before trying
machine-learning approaches, it’s useful to try simple, cheap machine-learning models (such as small, densely connected networks) before looking into complicated and
computationally expensive models such as RNNs. This is the best way to make sure any
further complexity you throw at the problem is legitimate and delivers real benefits.

The following listing shows a fully connected model that starts by flattening the
data and then runs it through two Dense layers. Note the lack of activation function on
the last Dense layer, which is typical for a regression problem. You use MAE as the loss.
Because you evaluate on the exact same data and with the exact same metric you did
with the common-sense approach, the results will be directly comparable.

Some of the validation losses are close to the no-learning baseline, but not reliably.
This goes to show the merit of having this baseline in the first place: it turns out to be
not easy to outperform. Your common sense contains a lot of valuable information
that a machine-learning model doesn’t have access to.

You may wonder, if a simple, well-performing model exists to go from the data to
the targets (the common-sense baseline), why doesn’t the model you’re training find it
and improve on it? Because this simple solution isn’t what your training setup is looking for. The space of models in which you’re searching for a solution—that is, your
hypothesis space—is the space of all possible two-layer networks with the configuration
you defined

These networks are already fairly complicated. When you’re looking for a solution with a space of complicated models, the simple, well-performing baseline may
be unlearnable, even if it’s technically part of the hypothesis space. That is a pretty significant limitation of machine learning in general: unless the learning algorithm is
hardcoded to look for a specific kind of simple model, parameter learning can sometimes fail to find a simple solution to a simple problem. 

#### 6.3.5 A first recurrent baseline

The first fully connected approach didn’t do well, but that doesn’t mean machine
learning isn’t applicable to this problem. The previous approach first flattened the
timeseries, which removed the notion of time from the input data. Let’s instead look
at the data as what it is: a sequence, where causality and order matter. You’ll try a
recurrent-sequence processing model—it should be the perfect fit for such sequence
data, precisely because it exploits the temporal ordering of data points, unlike the first
approach.

 Instead of the LSTM layer introduced in the previous section, you’ll use the GRU
layer, developed by Chung et al. in 2014.5
 Gated recurrent unit (GRU) layers work
using the same principle as LSTM, but they’re somewhat streamlined and thus
cheaper to run (although they may not have as much representational power as
LSTM). This trade-off between computational expensiveness and representational
power is seen everywhere in machine learning.

The new validation MAE of ~0.265 (before you start significantly overfitting) translates
to a mean absolute error of 2.35˚C after denormalization. That’s a solid gain on the
initial error of 2.57˚C, but you probably still have a bit of a margin for improvement. 

#### 6.3.6 Using recurrent dropout to fight overfitting

It’s evident from the training and validation curves that the model is overfitting: the
training and validation losses start to diverge considerably after a few epochs.

It has long been known that
applying dropout before a recurrent layer hinders learning rather than helping with
regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,6
 determined the proper way to use dropout with a recurrent network: the same
dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep.

What’s more, in order to regularize the representations formed by the recurrent gates
of layers such as GRU and LSTM, a temporally constant dropout mask should be applied
to the inner recurrent activations of the layer (a recurrent dropout mask). Using the
same dropout mask at every timestep allows the network to properly propagate its
learning error through time; a temporally random dropout mask would disrupt this
error signal and be harmful to the learning process.

 Yarin Gal did his research using Keras and helped build this mechanism directly
into Keras recurrent layers. Every recurrent layer in Keras has two dropout-related
arguments: dropout, a float specifying the dropout rate for input units of the layer,
and recurrent_dropout, specifying the dropout rate of the recurrent units

Because networks being regularized with dropout always take longer to fully converge, you’ll train the network for twice as many epochs.

#### 6.3.7 Stacking recurrent layers

Because you’re no longer overfitting but seem to have hit a performance bottleneck,
you should consider increasing the capacity of the network.

 it’s generally a good idea to increase the
capacity of your network until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As
long as you aren’t overfitting too badly, you’re likely under capacity.

Increasing network capacity is typically done by increasing the number of units in
the layers or adding more layers. Recurrent layer stacking is a classic way to build
more-powerful recurrent networks:

To stack recurrent layers on top of each other in Keras, all intermediate layers
should return their full sequence of outputs (a 3D tensor) rather than their output at
the last timestep. This is done by specifying return_sequences=True.

You can see that the added layer does improve the
results a bit, though not significantly. You can draw two conclusions:
 Because you’re still not overfitting too badly, you could safely increase the size of
your layers in a quest for validation-loss improvement. This has a non-negligible
computational cost, though.
 Adding a layer didn’t help by a significant factor, so you may be seeing diminishing returns from increasing network capacity at this point.

#### 6.3.8 Using bidirectional RNNs

The last technique introduced in this section is called bidirectional RNNs. A bidirectional RNN is a common RNN variant that can offer greater performance than a regular RNN on certain tasks. It’s frequently used in natural-language processing—you
could call it the Swiss Army knife of deep learning for natural-language processing

RNNs are notably order dependent, or time dependent: they process the timesteps
of their input sequences in order, and shuffling or reversing the timesteps can completely change the representations the RNN extracts from the sequence. This is precisely the reason they perform well on problems where order is meaningful, such as
the temperature-forecasting problem

A bidirectional RNN exploits the order sensitivity of RNNs: it consists of using two regular RNNs, such as the GRU and LSTM layers
you’re already familiar with, each of which processes the input sequence in one direction (chronologically and antichronologically), and then merging their representations. By processing a sequence both ways, a bidirectional RNN can catch patterns that
may be overlooked by a unidirectional RNN.

Remarkably, the fact that the RNN layers in this section have processed sequences in
chronological order (older timesteps first) may have been an arbitrary decision.

The reversed-order GRU strongly underperforms even the common-sense baseline,
indicating that in this case, chronological processing is important to the success of your
approach

This makes perfect sense: the underlying GRU layer will typically be better at
remembering the recent past than the distant past, and naturally the more recent
weather data points are more predictive than older data points for the problem (that’s
what makes the common-sense baseline fairly strong). Thus the chronological version
of the layer is bound to outperform the reversed-order version. 

Importantly, this isn’t
true for many other problems, including natural language: intuitively, the importance
of a word in understanding a sentence isn’t usually dependent on its position in the sentence

A bidirectional RNN exploits this idea to improve on the performance of chronological order RNNs. It looks at its input sequence both ways (see figure 6.25), obtaining potentially richer representations and capturing patterns that may have been missed by the
chronological-order version alone.

#### 6.3.9 Going even further

There are many other things you could try, in order to improve performance on the
temperature-forecasting problem:
 Adjust the number of units in each recurrent layer in the stacked setup. The
current choices are largely arbitrary and thus probably suboptimal.
 Adjust the learning rate used by the RMSprop optimizer.
 Try using LSTM layers instead of GRU layers.
 Try using a bigger densely connected regressor on top of the recurrent layers:
that is, a bigger Dense layer or even a stack of Dense layers.
 Don’t forget to eventually run the best-performing models (in terms of validation MAE) on the test set! Otherwise, you’ll develop architectures that are overfitting to the validation set.

As always, deep learning is more an art than a science. We can provide guidelines that
suggest what is likely to work or not work on a given problem, but, ultimately, every
problem is unique; you’ll have to evaluate different strategies empirically. There is
currently no theory that will tell you in advance precisely what you should do to optimally solve a problem. You must iterate

#### 6.3.10 Wrapping up

Here’s what you should take away from this section:
 As you first learned in chapter 4, when approaching a new problem, it’s good to
first establish common-sense baselines for your metric of choice. If you don’t
have a baseline to beat, you can’t tell whether you’re making real progress.
 Try simple models before expensive ones, to justify the additional expense.
Sometimes a simple model will turn out to be your best option.
 When you have data where temporal ordering matters, recurrent networks are
a great fit and easily outperform models that first flatten the temporal data.
 To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the dropout and recurrent_dropout arguments
of recurrent layers.
 Stacked RNNs provide more representational power than a single RNN layer.
They’re also much more expensive and thus not always worth it. Although they
offer clear gains on complex problems (such as machine translation), they may
not always be relevant to smaller, simpler problems.
 Bidirectional RNNs, which look at a sequence both ways, are useful on natural language processing problems. But they aren’t strong performers on sequence
data where the recent past is much more informative than the beginning of the
sequence.

### 6.4 Sequence processing with convnets

In chapter 5, you learned about convolutional neural networks (convnets) and how
they perform particularly well on computer vision problems, due to their ability to
operate convolutionally, extracting features from local input patches and allowing for
representation modularity and data efficiency. The same properties that make convnets excel at computer vision also make them highly relevant to sequence processing.
Time can be treated as a spatial dimension, like the height or width of a 2D image.

Such 1D convnets can be competitive with RNNs on certain sequence-processing
problems, usually at a considerably cheaper computational cost. Recently, 1D convnets, typically used with dilated kernels, have been used with great success for audio
generation and machine translation. In addition to these specific successes, it has long
been known that small 1D convnets can offer a fast alternative to RNNs for simple tasks
such as text classification and timeseries forecasting

#### 6.4.1 Understanding 1D convolution for sequence data

you can use 1D convolutions, extracting local 1D patches (subsequences) from sequences

Such 1D convolution layers can recognize local patterns in a sequence. Because the
same input transformation is performed on every patch, a pattern learned at a certain
position in a sentence can later be recognized at a different position, making 1D convnets translation invariant

. For instance, a 1D convnet processing sequences of characters using convolution windows of size 5 should be able to
learn words or word fragments of length 5 or less, and it should be able to recognize these words in any context in an input sequence

A character-level 1D convnet is thus
able to learn about word morphology.

#### 6.4.2 1D pooling for sequence data

You’re already familiar with 2D pooling operations, such as 2D average pooling and
max pooling, used in convnets to spatially downsample image tensors. The 2D pooling
operation has a 1D equivalent: extracting 1D patches (subsequences) from an input
and outputting the maximum value (max pooling) or average value (average pooling).
Just as with 2D convnets, this is used for reducing the length of 1D inputs (subsampling). 

#### 6.4.3 Implementing a 1D convnet

In Keras, you use a 1D convnet via the Conv1D layer, which has an interface similar to
Conv2D. It takes as input 3D tensors with shape (samples, time, features) and
returns similarly shaped 3D tensors. The convolution window is a 1D window on the
temporal axis: axis 1 in the input tensor.

1D convnets are structured in the same way as their 2D counterparts, which you used
in chapter 5: they consist of a stack of Conv1D and MaxPooling1D layers, ending in
either a global pooling layer or a Flatten layer, that turn the 3D outputs into 2D outputs, allowing you to add one or more Dense layers to the model for classification or
regression.

One difference, though, is the fact that you can afford to use larger convolution
windows with 1D convnets. With a 2D convolution layer, a 3 × 3 convolution window
contains 3 × 3 = 9 feature vectors; but with a 1D convolution layer, a convolution window of size 3 contains only 3 feature vectors. You can thus easily afford 1D convolution
windows of size 7 or 9.

#### 6.4.4 Combining CNNs and RNNs to process long sequences

Because 1D convnets process input patches independently, they aren’t sensitive to the
order of the timesteps (beyond a local scale, the size of the convolution windows),
unlike RNNs. Of course, to recognize longer-term patterns, you can stack many convolution layers and pooling layers, resulting in upper layers that will see long chunks of
the original inputs—but that’s still a fairly weak way to induce order sensitivity. 

One
way to evidence this weakness is to try 1D convnets on the temperature-forecasting
problem, where order-sensitivity is key to producing good predictions.

 the convnet looks for patterns anywhere in the input timeseries and has no knowledge of the temporal position of a pattern it sees 
 
. Because more recent
data points should be interpreted differently from older data points in the case of this
specific forecasting problem, the convnet fails at producing meaningful results.

One strategy to combine the speed and lightness of convnets with the order-sensitivity
of RNNs is to use a 1D convnet as a preprocessing step before an RNN (see figure 6.30).
This is especially beneficial when you’re dealing with sequences that are so long they can’t
realistically be processed with RNNs, such as
sequences with thousands of steps.

The convnet will turn the long input sequence into
much shorter (downsampled) sequences of
higher-level features. This sequence of
extracted features then becomes the input to
the RNN part of the network.

This technique isn’t seen often in
research papers and practical applications,
possibly because it isn’t well known. It’s effective and ought to be more common.

#### 6.4.5 Wrapping up

 In the same way that 2D convnets perform well for processing visual patterns in
2D space, 1D convnets perform well for processing temporal patterns. They
offer a faster alternative to RNNs on some problems, in particular naturallanguage processing tasks.
 Typically, 1D convnets are structured much like their 2D equivalents from the
world of computer vision: they consist of stacks of Conv1D layers and MaxPooling1D layers, ending in a global pooling operation or flattening operation.
 Because RNNs are extremely expensive for processing very long sequences, but
1D convnets are cheap, it can be a good idea to use a 1D convnet as a preprocessing step before an RNN, shortening the sequence and extracting useful representations for the RNN to process. 

### 7 Advanced Deep-Learning Best Practices

This chapter explores a number of powerful tools that will bring you closer to
being able to develop state-of-the-art models on difficult problems. Using the Keras
functional API, you can build graph-like models, share a layer across different
inputs, and use Keras models just like Python functions.

Keras callbacks and the
TensorBoard browser-based visualization tool let you monitor models during training. We’ll also discuss several other best practices including batch normalization,
residual connections, hyperparameter optimization, and model ensembling.

#### 7.1 Going beyond the Sequential model: the Keras functional API

The Sequential model makes the assumption that the
network has exactly one input and exactly one output, and
that it consists of a linear stack of layers

 But this set of
assumptions is too inflexible in a number of cases. Some
networks require several independent inputs, others
require multiple outputs, and some networks have internal branching between layers that makes them look like
graphs of layers rather than linear stacks of layers

ome tasks, for instance, require multimodal inputs: they merge data coming from
different input sources, processing each type of data using different kinds of neural
layers. Imagine a deep-learning model trying to predict the most likely market price of
a second-hand piece of clothing, using the following inputs: user-provided metadata
(such as the item’s brand, age, and so on), a user-provided text description, and a picture of the item. 

A
naive approach would be to train three separate models and then do a weighted average of their predictions. But this may be suboptimal, because the information
extracted by the models may be redundant. A better way is to jointly learn a more accurate model of the data by using a model that can see all available input modalities
simultaneously: a model with three input branches 

Similarly, some tasks need to predict multiple target attributes of input data. Given the
text of a novel or short story, you might want to automatically classify it by genre (such
as romance or thriller) but also predict the approximate date it was written. Of course,
you could train two separate models: one for the genre and one for the date. But
because these attributes aren’t statistically independent, you could build a better
model by learning to jointly predict both genre and date at the same time.

Such a
joint model would then have two outputs, or heads (see figure 7.3). Due to correlations between genre and date, knowing the date of a novel would help the model
learn rich, accurate representations of the space of novel genres, and vice versa.

There’s also the
recent trend of adding residual connections to a model, which started with the ResNet
family of networks 

These three important use cases—multi-input models, multi-output models, and
graph-like models—aren’t possible when using only the Sequential model class in
Keras. But there’s another far more general and flexible way to use Keras: the functional API. This section explains in detail what it is, what it can do, and how to use it.