# Guideline of Machine Learning for Tabular Data

In this section we talk about tricks and tips for data analysis with machine learning. 

All the nodebooks in this section:

* Reggression:

  * [Boston House Price Prediction]()

* Classification:

  * [Cancer Prediction]()

Check the following tips to examine your data and improve your model!

## Data Collection (General Guide)

Please check here for data collection:

* What do you want to predict
* What and where to get your data
* If you need need ETL pipeline, check here: 

## Exploratory Data Analysis (EDA)

You should know your data, here is what you can do for your data:

* Data Cleansing
   * Outlier Detection 
* Data Visualization
   * Data Distribution Analysis
       * Only models like LDA, Gaussian Naive Bayes, Logistic Regression, Linear Regression assume Normality. For models like decision tree, it is better to have even amounts of data for a feature.
    * Scatter Plot
    * Correlation Plot
* Principle Component Analysis (PCA)
* Feature Engineering
* For classification task, is the amount of data for each class even? do you need oversampling or undersampling?
* Data Augmentation
   * DeltaPy⁠⁠ [`github`](https://github.com/firmai/deltapy)

## Model Selection and Training

### Model Ensemble

* Voting 
* Stacking

### Transformer for Tabular Data Analysis

* TabNet [`github`](https://github.com/google-research/google-research/tree/master/tabnet)

### AutoML for Tabular Data Analysis

* AutoGluon [`github`](https://github.com/awslabs/autogluon)

## Model Evaluation, Inspection and Debugging

### Cross-Validation

### Model Explanation

Models like Linear Regression and Decsion Tree has better explanbility. However, 


## Tools

### Data Validation

* Facets [`github`](https://github.com/PAIR-code/facets)

### Quick Model Testing

* Lazy Predict [`github`](https://github.com/shankarpandala/lazypredict)

### Explainity

* SHAP [`github`](https://github.com/slundberg/shap)
* Manifold [`github`](https://github.com/uber/manifold)

### Debugging

* Tensorflow Model Analysis

* Microsoft Responsible AI Widget


### Data Visualization

### Data Versioning

* DVC
* Pachyderm

### Model Tracking


## Interesting Research for Auto Data Analysis

* [AutoVis]()
* [Auto EDA]()
* [AutoML]()


