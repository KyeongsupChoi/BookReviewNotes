# Machine Learning for Beginners

| Title            | Machine Learning for Beginners |
|------------------|--------------------------------|
| Author           | Oliver Theobald                |
| Publication Date | 2021                           |
| Page Count       | 179                            |
| Genre            | Non-Fiction                    |
| ISBN             | 979-8558098426                 |
| Date Read        | 2023-02-07                     |
| Date Finished    | 2023-02-22                     |

## Description

Machine Learning Resource Book

## Summary

### Chapter 1: Preface

-This book focuses on high-level fundamentals, including key terms, general workflow, and the statistical underpinnings of basic algorithms.

### Chapter 2: What is Machine Learning?

__Machine Learning:__ A subfield of computer science that gives computers the ability to learn without being explicitly programmed.  

__Self Learning:__ The application of statistical modeling to detect patterns and improve performance based on data and empirical information.  

-Machine learning is based on input data to function rather than input commands.

__Hyperparameters:__ The parameters that are explicitly defined to control the learning process.  

-Exposure to input data thereby deepens the model's understanding of patterns, including the signicant changes in the data, and to construct an effective self-learning model.  

__Training Data:__ The initial reserve of data used to train the machine learning model.  

__Test Data:__ The reserve of data used to evaluate the algorithm or training model.  

-Computer Science > Data Science > AI > Machine Learning

-Whereas data mining focuses on analyzing input variables to predict a new output, machine learning extends to analyzing both input and output variables.  

-In unsupervised learning only the input is known and in reinforcement learning only the output is known.

### Chapter 3: Machine Learning Categories

-Supervised learning requires known input(labeled dataset) and includes regression analysis(i.e. linear regression, logisitc regression, non-linear regression) decision trees, k-nearest neighbors, neural networks and support vector machines.  

-Unsupervised learning instead focuses on analyzing relationships between input variables and uncovering hidden patterns that can be extracted to create new labels regarding possible outputs. 

-Unsupervised learning is hard to validate and can be subjective.  

-Examples of unsupervised learning algorithms include social network analysis and descending dimension algorithms.  

-Semi-supervised learning hybridizes labeled and unlabeled cases with the goal of leveraging unlabeled cases to improve the reliability of the prediction model.  

-Reinforcement learning builds its prediction model by gaining feedback from random trial and error and leveraging insight from previous iterations.  

-A specific example of a reinforcement learning algorithm is Q-learning. 

### Chapter 4: The Machine Learning Toolbox

-Data constitutes the input needed to train the model and generate predictions.  

-In a tabular dataset the columns would be a feature/variable/dimension/attribute and the rows would be a case/value.  

-Machine learning infrastructure consists of platforms and tools for processing data. 

-The central tools are the machine learning algorithms.  

-Data visualization is used to communicate results to relevant decision-makers.  

-Advanced machine learning uses, big data techniques, cloud infrastructure, and advanced algorithms.  

### Chapter 5: Data Scrubbing  

-Data scrubbing is the technical process of refining the dataset to make it more workable.  

-Feature selection is the identification of which variables are the most relevant to the objective or model. 

-Row compression merges rows to reduce the total number of data points.  

-One hot encoding converts values into binary form as text-based values are not compatible with most algorithms.  

-Binning is a method of feature engineering used for converting continuous numeric values into multiple binary features called bins or buckets according to their range of values.  

-Normalization rescales the range of values into a set range with a prescribed minimum or maximum, not recommended for features with an extreme range.  

-Standardization converts unit variance to a standard normal distribution with a mean of zero and a standard deviation of one.  

-Missing data can be replaced with the mode or median values or removed altogether.  

### Chapter 6: Setting up your Data 

-Split validation is splitting the data into two segments for training and testing, usually 80/20, 70/30.  

-Randomizing the row order helps to avoid bias in the model.  

-It is imperative not to test your model with the same data you used for training.  

-Classification tasks use performance metrics such as Area Under Curve(AUC), Receiver Operating Characteristics(ROC), confusion matrix, recall, and accuracy.

-Models that provide a numeric output use performance metrics such as mean absolute error and root mean square error.  

-If the average MAE or RMSE is much higher using the test data than the training data, this is usually an indication of overfitting in the model.  

-Hyperparameters control and impact how fast the model learns patterns and which patterns to identify and analyze.  

-Cross validation maximizes the availability of training data by splitting data into various combinations and testing each specific combination.  

-Exhaustive cross validation involves finding and testing all possible combinations to divide the original sample into a training set and test set.  

-K-fold validation divides the data into a k number of buckets and tests all combinations of the buckets as a training set made of k-1 buckets and one test set bucket. 

-At an absolute minimum, a basic machine learning model should contain 10 times as much data points as the total number of features. 

### Chapter 7: Linear Regression

-Linear regression generates a straight line to describe linear relationships.  

-The goal of linear regression is to split the data in a way that minimizes the distance between the hyperplane and the observed values.  

-The distance between the best fit line and the observed values is called the residual or error.  

-Multiple linear regression is linear regression with more than one independent variable.  

-While the output of linear regression must be continuous in the form of a floating-point or integer value the input can be continuous or categorical.  

-Multi-collinearity happens when a strong linear correlation exists between two independent variables and cancels each other out in the model.  

-To avoid multi-collinearity the relationship between each combination of independent variables need to be checked.  

### Chapter 8: Logistic Regression

-Logistic regression is a supervised learning technique that produces a qualitative prediction, usually binary prediction.  

-The logistic hyperplane represents a classification/decision boundary rather than a prediction trendline. 

-Multinomial logistic regression solves multiclass problems with more than two possible discrete outcomes.  

-In general, logistic regression normally doesn't work so well with large datasets and especially messy data containing outliers, complex relationships and missing values.  

### Chapter 9: k-Nearest Neighbors 

-K-NN is a supervise learning algorithm that classifies new data points based on their position to nearby data points.  

-K is the number of nearest data points used to classify the new data point, five is the default, an even number can lead to a stalemate.  

-This algorithm works best with continuous variables but it is still possible to use binary categorical variables represented as 0 and 1.  

-K-NN is computationally expensive and even more so with multiple dimensional data. 

### Chapter 10: k-Means Clustering

-K-means clustering is an unsupervised learning technique that involves grouping or clustering data points that share similar attributes.  

-K is the number of discrete groups or clusters.  

-K number of centroids are manually selected, then the Euclidean distance of each data point from the centroids are calculated and the minimum determines the cluster. Then a new centroid is calculated based on the cluster's mean value and repeated until the data points no longer switch clusters. 

-Scree plot charts can be used to decide K.  

-Square root of number of data points divided by 2 is also used to decide K.  

-Domain knowledge can also be used to determine K.  

### Chapter 11: Bias & Variance 

-Bias refers to the gap between the value predicted by the model and the actual value of the data.  

-Variance describes how scattered the predicted values are in relation to each other.  

-Underfitting results in low variance and high bias.  

-Overfitting results in high variance or low bias.  

-Overfitting can occur if the training and testing data are not randomized.  

-Underfitting can occur if the model is overly simple or has insufficient training data.  

-Increasing K in K-NN or switching from a single decision tree to random forests with many decision trees can reduce variance.  

-Regularization reduces the risk of overfitting by constraining the model to make it simpler, but it artificially amplifies bias error.  

-Cross validation improves model accuracy by minimizing pattern discrepancies between the training and testing data.  

### Chapter 12: Support Vector Machines

-Support Vector Machines(SVM) are mostly used as a classification technique for predicting categorical outcomes at a position of maximum distance from the data points.  

-The margin is the distance between the decision boundary and the nearest data point multiplied by two.  

-SVM is used to untangle complex relationships and mitigate outliers and anomalies.  

-The C value of the margin can be used to adjust accuracy.  

-The kernel trick helps classify high-dimensional data by mapping to a 3 dimensional space.  

### Chapter 13: Artificial Neural Networks  

-Artificial neural networks is a popular machine learning technique that utilized nodes, edges, and weights.  

-Each edge in the network has a numeric weight that can be altered based on experience. If the sum of the connected edge satisfies a set threshold known as the activation function, it activates a neuron at the next layer.  

-The cost or cost value is the difference between the predicted output and the actual output.  

-Back-propogation is training the neural network to achieve the lowest cost value.  

-Neural networks are black boxes with limited to no insight about how specific variables influences its decision.  

-Neural networks generally fit prediction tasks with a large number of input features and complex patterns.  

-The middle layers are considered hidden because they covertly process objects between the input and output layers. As more hidden layers are added to the network, the model's capacity to analyze complex patterns improves.

-Deep learning usually has at least 5-10 node layers.  

-Advanced deep learning techniques include multiplayer perceptions(MLP), convolution networks, recurrent networks, deep belief networks, and recursive neural tensor networks(RNTN).  

### Chapter 15: Decision Trees    

-A neural network requires a hug amount of input data and computational resources and also are a black box.  

-Decision trees are easy to interpret, work with less data, and consume less computational resources.  

-Decision trees are used primarily for solving classification problems but can also be used as a regression model to predict numeric outcomes.  

-Splits are edges and leaves are nodes/decision points/terminal nodes.  

-Entropy is a mathematical concept that explains the measure of variance in the data among the classes.  

-A decision tree should reduce entropy at each layer.  

-Decision trees are particularly vulnerable to overfitting.  

-Bagging involves growing multiple decision trees using a randomized selection of input data for each tree and combining the result. 

-Bootstrap sampling extracts a random variation of the data at each round.  

-Random forests limit the choice of variables by capping the number of variables considered for each split.  

-Boosting creates new decision trees from misclassified cases in the previous tree by adding weights.  

-Gradient boosting selects variables that improve prediction accuracy with each tree. 

### Chapter 16: Ensemble Modeling  

-Ensemble modeling combines algorithms or models to build a unified prediction model.  

-Ensemble modeling covers the weaknesses of single models.  

-Bagging draws upon randomly drawn data and combines predictions to design a unified model.  

-Boosting is a homogenous ensemble that produces a sequential model.  

-Stacking runs multiple models simultaneously on the data and combines those results to produce a final model.  
