# Predictive-Modeling-for-Heart-Attacks
Columbia Data Science Final Project
Group Project with Joan Akibode, Sanaa Mouzahir, Minh Nguyen, Andrea Lin

### 1. Introduction
According to the Center for Disease Control and Prevention, heart disease is a leading cause of death within the United States [1]. Within heart diseases, heart attacks are one of the most common. In the United States, a person has a heart attack every forty seconds [2]. The goal is to take preventative measures to avoid heart attacks. In order to take preventative measures, features that lead to the cause of heart attacks must be identified. 

This project intends to explore various models that can sufficiently predict if a person is prone to heart attack or not. A set of features for categorical and continuous data will be used to identify a binary output of whether someone has a heart attack, where the classification of 0 denotes the normal state of no heart attack and 1 denotes the abnormal state of heart attack. The categorical datasets used in the models are number of major vessels, chest pain, fasting blood sugar, and thallium stress test results. The continuous variables are age, maximum heart rate, resting blood pressure, cholesterol level, and previous peak. 

### 2. Methodology
#### 2.1. Tools
To carry out the data analysis, Python programming language was used in the Google Colaboratory environment. Python’s strength lies in the availability of libraries that can efficiently carry out functions common to data science and machine learning processes. Aside from the common Python data wrangling libraries such as NumPy, Pandas, the main library that was utilized for carrying out machine learning algorithms is the Scikit Learn library.
#### 2.2. Data Preprocessing 
##### 2.2.1. Exploratory Data Analysis
Characteristics of the dataset were initially explored through plots (shown in Figure 1) and descriptive statistics to better understand how to handle the data as well as determine the distribution of the data. For the input variables, ensuring normal distribution of continuous variables and checking the frequency of categorical variables can help ensure stability in the algorithms to be used in modelling. For output variables, ensuring a balanced class split is important because most modeling algorithms were designed for even class splits and if there is imbalance, then it must be handled accordingly with specialized techniques. Figure 2 shows a balanced split between the two classes under consideration.


##### 2.2.2. Feature selection 
Pandas Profiling is a tool that generates a report of data statistics, which was used to study the correlation within features and with the outcome. Figure 3 shows a correlation matrix produced by the tool, representing the weight of correlation measured with Spearman’s coefficient (since the dataset contains discrete variables and conventional Pearson’s coefficient cannot be used). 
When multiple variables were correlated with each other, the feature most correlated with the output was kept or, in certain cases, the more relevant feature was kept. Ultimately, the variables that were dropped were: exercise, sex, slope, and resting electrocardiographic results. The features most correlated with heart attack were chest pain, maximum heart rate, exercise and thallium stress test results. Some of these correlation relationships are intuitive, but other results, such as the lack of correlation between age and heart attack, were surprising. Aside from the necessity of removing collinearity, this further reinforces the value of correlation analysis.
##### 2.2.3. Data Split
With only one set of data to build and validate the predictive model, a split was performed to use one set of data for training the model and the other set for testing the model. A conventional split ratio was used, with 80% for the training set and 20% for the testing set. 
##### 2.2.4. Data Scaling
Standardization of data for continuous variables can help ensure stability in certain models, so a standard scaler from scikit-learn is applied to our data after determining that the data follows a normal distribution. 

#### 2.3. Model Selection
##### 2.3.1. Logistic regression
The first model utilized is the classic logistic regression algorithm, which considers probability scores for binary classification problems such as this one, where a threshold can be chosen to predict the outcome of 0 or 1. 
##### 2.3.1.1. Model Rationale 
The features in this dataset qualify for logistic regression because they are linear and independent after removing collinearity. Additionally, the amount of features in this dataset is not greater than the number of samples, which is an important consideration when using logistic regression.
##### 2.3.1.2. Model Hyperparameters 
Two trials of logistic regression were performed, one using the Statsmodel library and the other using Scikit-learn library with bootstrapping introduced in an attempt to increase prediction accuracy. Bootstrapping is a method of repeatedly selecting from the original data a set of  randomly shuffled samples to perform logistic regression on, and then the mean, standard error and distribution of the combined results can be retrieved. Table 1 and Figure 4 shows the more  complex results of model parameters that can be derived from bootstrapping. 

Logistic regression is prone to overfitting and a penalty term can be introduced to address this. However, this is not necessary here as the data produces a simple model and the risk of overfitting is not extreme. No other hyperparameters are available for adjusting this algorithm.

##### 2.3.2. Support Vector Machine 
###### 2.3.2.1. Model Rationale
The second model utilized is the Support Vector Machine (SVM), which is another algorithm that is useful for classification problems. SVM can be highly accurate with low computational requirements and provides greater flexibility in hyperparameter tuning. In this case where data can clearly be separated, SVM can work well.
###### 2.3.2.2. Model Hyperparameters
The flexibility of SVM allows for nonlinear classification, and kernels of different dimensions are used for this purpose. In the case of this data, a linear kernel is chosen for simplicity. The model can also be tuned with the hyperparameters C , which affects the strength of the regularization, consequently controlling the margin of the classifier. A cross validation grid-search from Scikit-Learn is performed to find the optimal value for C, chosen from a range of 1 to 10, with 5 being the final optimal value chosen.
##### 2.3.3. Random Forest 
###### 2.3.3.1. Model Rationale
The final model utilized is the Random Forest algorithm, an ensemble method built on the foundation of decision trees but allows for variance reduction through a combination of bootstrapping and aggregation (termed “bagging”). 
###### 2.3.3.2. Model hyperparameters
In Random Forest modeling, various hyperparameters associated with formation of the decision trees as well as bagging strategies are available. In this model, the maximum depth of the trees and the number of estimators (i.e., number of trees) are the parameters that were explored. A cross validation grid search is performed to find the optimal values for these parameters. The optimal number of parameters are chosen from [12,13,14,15], and the optimal maximum depth is chosen from [3,4,5,6]. From those options, the optimal values for number of estimators and maximum depth are 13 and 4, respectively.

### 3. Results
Accuracy scores, Receiver Operating Characteristic (ROC) curves, and confusion matrices are used to represent the performance of the three predictive models. The logistic regression model had an accuracy score of 87%, the SVM of 85%, and the Random Forest of 82%. Figure 5 shows the ROC curves for each of the models representing the ratio of the false positive rate to the true positive rate. Figure 6 displays the confusion matrices for each of the models, where the true positives are shown in the light pink and the false negatives are shown in the dark red color. Overall, the logistic regression model yielded the best results.

### 4. Discussion 
#### 4.1. Significance of Results 
While accuracy score provides a standard for comparing the models, the ROC curves and confusion matrices are particularly relevant in evaluating the performance of the models in the context of heart attack predictions. The Type I error of a false positive of possible heart attack is predicted for a given health characteristic might result in resource waste, whereas the Type II error where a false negative is predicted means a lost opportunity to alert an at-risk individual. Preventative measures and extra health monitoring may be instrumental in avoiding deadly heart attacks. Under this consideration, the ROC curve may not be as important to examine as the false negative rates shown in the confusion matrices. SVM shows a better ROC curve than the Random Forest model, but a higher false negative score in the confusion matrix. While the Random Forest model had a lower accuracy score than the Logistic Regression model, they had even false negative rates. The decision to weigh Type 1 or Type II error as more important may change under different operating conditions, however. Thus the results should be examined accordingly and improvements to the model should take this into consideration.

#### 4.2. Recommendations for Model Improvement
In the first model used to train the data, logistic regression was the obvious first choice for this binary classification problem. The resulting accuracy score of 87% is strong but there is room for improvement. The limitation of this dataset may be the number of samples, since logistic regression prefers larger sample sizes. Due to the limited number of samples, the attempt to introduce bagging to the logistic regression model did not yield results significantly different from the simple model. With few other hyperparameters to tune in logistic regression, the next step was to explore other algorithms for binary classification. Thus, the SVM and Random Forest models were pursued. 

In this initial exploration, the full flexibility of SVM and tunability of random forest was not explored, thus the results for these two models may be further improved with more thorough examination of hyperparameter tuning. This could include the consideration of more hyperparameters along with the use of different methods of hyperparameter optimization. In this work, cross validation grid search was used, which takes a set of predefined hyperparameter inputs to search from, as outlined in methodology. Other methods such as random search or Bayesian hyperparameter optimization may uncover better parameters for enhanced model performance. 

Additionally, obtaining more data can also help improve the model accuracy and explore other areas that may affect the higher risk in heart diseases. For example, data about the life-style of a person, such as diet, average sleep hours, etc. can be obtained to see a different point of view of the study. Other important data such as weights and family history that were left out may also be helpful in improving the prediction of heart diseases. Then, with the improvement of the models, these models can be used to predict cases of other health related issues. The architecture of the models are very generic and can easily be modified to apply to other problems. 



### References

[1] Centers for Disease Control and Prevention. Underlying Cause of Death, 1999–2018. CDC WONDER Online Database. Atlanta, GA: Centers for Disease Control and Prevention; 2018. Accessed March 12, 2020.

[2] Fryar CD, Chen T-C, Li X. Prevalence of uncontrolled risk factors for cardiovascular disease: United States, 1999–2010 pdf icon[PDF-494K]. NCHS data brief, no. 103. Hyattsville, MD: National Center for Health Statistics; 2012. Accessed May 9, 2019.


