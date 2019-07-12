# STA 208 Project

## Application of Classification Methods in Analyzing and Predicting Loan Status

J. Bian,  Y. Du,   D. Peng


data source: https://www.lendingclub.com/info/download-data.action

The files are supposed to read in this order:   
1. [Variable Selection and Data Description](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Variable%20Selection%20and%20Data%20description.ipynb)  
2. [PCA, KNN and Random Forest Classifiers](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/PCA%2C%20KNN%20and%20Random%20Forest%20Clsifier.ipynb)  
3. [SVM](SVM.ipynb)
4. [Logistic Regression and Prediction in Reality](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Logistic%20Regression%20and%20Prediction%20in%20Reality.ipynb)  
5. [Prediction Using Random Forest](PredictionUsingRandomForest.ipynb)


## 1. Introduction

First we download loan data in 2016 from LendingClub. After preliminary data cleansing, we keep 65 variables.

## 2. Variable Selection and Data Discription

### 2.1 Variable Selection Using Group Lasso
Since lot of similar information are included among the 64 predictor variables, and the categorical variables would be added in as groups, we apply Group Lasso in this case and you can find the steps at [2.3 Variable Selection (Group Lasso)](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Variable%20Selection%20and%20Data%20description.ipynb)

### 2.2 Numerical Variables Description
In this part we explore relationship among selected numerical predictor variables(computing correlation matrix and drawing heatmap). Relationship between loan status and some predictor variables we are interested in is also analyzed.(You are able to see the analysis procedure and results at [Variable Selection and Data Description 2.4.1 Numerical](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Variable%20Selection%20and%20Data%20description.ipynb)

### 2.3 Categorical Variables Description
For  selected  dummy  variables,  counter  barplot  for  pairs  of  each  categorical  variable  and  loan  status  were draw sequentially. (You are able to see the analysis procedure and results at [Variable Selection and Data Description 2.4.2 Categorical](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Variable%20Selection%20and%20Data%20description.ipynb)

### 2.4 Principal Component Analysis
We use Principal Component Analysis(PCA) method to obtain a clearer interpretation of the differences between borrowers that lead to different loan status. The number of principal components is decided as 10, and a Varimax Rotation is implemented to recogonize the most powerful components. Details in [PCA](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/PCA%2C%20KNN%20and%20Random%20Forest%20Clsifier.ipynb)

## 3. Model Building and Selection

### 3.1 Support Vector Machines
In this part SVM is chosed to do calssification of loan status. Results show that test error is 0.0338 and accuracy score is 0.9661 , which means our model fits the data well. Moreover, ROC area is 0.87, indicating high prediction accuracy and PR curve also shows high precision and high recall value.(You are able to see the analysis procedure and results at [SVM](SVM.ipynb))


### 3.2 Logistic Regression  
Logistic Regression is applied here as it is a classical model for binary response variable. You can see the procedure and conclusions at [Logistic Regression and Prediction in Reality](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Logistic%20Regression%20and%20Prediction%20in%20Reality.ipynb).

### 3.3 KNN Classification
K-nearest-neighbor Classifier is also applied to predict the loan status. The tuning parameter k (number of neighbors) is determined by computing the misclassification rate on the testing set by fitting the classifier on the training set. The test error picks k = 8. And the model predicts about 95% of the data right. Details in [KNN](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/PCA%2C%20KNN%20and%20Random%20Forest%20Clsifier.ipynb)

### 3.4 Random Forest Classification
We implement the random forest classifier with the number of trees prespecified as 1000, and compute the misclassification rate of prediction on the testing set. The accuracy rate is over 97%. Details in [random Forest](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/PCA%2C%20KNN%20and%20Random%20Forest%20Clsifier.ipynb)

## 4. Conclusion
### 4.1 Model Decision
According to prediction accuracy on testing set of four fitted models, __Random Forest Classifier__ has highest accuracy, which predicts 97.1% of the data to be in good loan status. Then we apply this model to future prediction in reality.

### 4.2 Future Prediction
We use above models to do prediction in reality. However, results are not so good as expected. __Random Forest Classifier__ only predicts about 30% as in good status and __SVM__ predicts 67%. Here __logistic__ preforms best as there are about 75% good-status loans. We also explore why does it happen, whose result can be seen at  
[Prediction Using Random Forest](PredictionUsingRandomForest.ipynb)   

[Prediction in Reality](SVM.ipynb)  

[Logistic](https://github.com/dandipeng/machinelearning/blob/master/Projects/Application%20of%20Classification%20Methods%20in%20Analyzing%20and%20Predicting%20Loan%20Status/Logistic%20Regression%20and%20Prediction%20in%20Reality.ipynb)

### 4.3 Improvement
As far as we concerned, if we want to predict status about recent loans, we should eliminate those variables containing mobile information, say, depend much on time for example. This is what we can make improvements in the future. Our model here is more suitable to do prediction upon more previous loans. However, we still have good reason to believe that this is a very interesting case study since we not only get deep understanding of different classification methods but also apply them into reality and get interesting results.
