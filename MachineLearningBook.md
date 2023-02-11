# Machine Learning Book

| Title            | Machine Learning for Beginners |
|------------------|--------------------------------|
| Author           | Oliver Theobald                |
| Publication Date | 2021                           |
| Page Count       | 179                            |
| Genre            | Non-Fiction                    |
| ISBN             | 979-8558098426                 |
| Date Read        | 2023-02-07                     |
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

-