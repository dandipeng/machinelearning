## Image Classification: Quick Draw! Doodle Recognition

J. Bian,   Y. Du,   S. Duan,   S. Jiang,   J. Li,   W. Lin,   D. Peng,  S. Xiao,   X. Xie,   Y. Zhu,     


### Introduction
Quick Draw is an experimental game to demonstrate the public how Google AI recognizes pictures, which asks users to draw an image depicting a specific category, such as "cat", "mouse" etc., and Google AI predicts this category by viewing the hand drawing.

The Quick Draw Dataset has millions of drawings across 300+ categories. However, in this project, we only aim at animal categories including ant, bear, bee, cat, crab, dragon, elephant, mouse, sea turtle, and snail. Our goal is to explore different models such as KNN, GBDT Boosting, Random Forest, SVM, and CNN to test which model performs the best.

The best classifier of each model was developed using cross-validation and area under PR and ROC curve.  Among best models of each methods, CNN has the highest accuracy: 86.25\%.

#### What is our data look like?

![Data Overview](images/EDA1.png)


### Method 1: KNN

K-nearest neighbor (KNN) finds similar instances around the target instance. By the majority voting around the "k" nearest neighbors, KNN predicts the class to the target instance. This method has advantages in a large number of classification, such as handwritten digits. Also, as a non-parametric method , KNN is good at the classification where the decision boundary is irregular. The parameters chosen to be tuned is n\_neighbors (Number of neighbors to use), with other parameters following the default settings.

The number of neighbors is determined by cross validation search( 3-fold search). Based on accuracy, the best number of neighbors is 13.  Here are the classification results for 13-NN model.

### Method 2: Random Forest

Random forest is another method could be used in image classification. It is an ensemble method based on the combination of multiple decision trees, and use the classification that most of the trees made as the final result. It runs efficiently on large data bases and provides excellent accuracy comparing with most of algorithms.

We tune the parameters with cross-validation(cv = 3) to find the best model and compare the results with the default model in ''sklear'' package.

**Parameters:**

* n\_estimators(the number of decision trees): usually the higher the number of trees the better to learn the data while adding a lot of trees can slow down the training process a lot. We decide to choose a relative large range of the parameter space: [100, 200, ......1000].

*  max\_features(max number of features considered for splitting a node): we choose the space as [`auto', `sqrt'].

*  max\_depth(max number of levels in each decision tree): usually the deeper the tree, the more information it captures about the data. We choose the space as [10, 20, ......110].

* min\_sample\_split(min number of data points placed in a node before the node is split): we choose the space as [2, 5, 10].

* min\_samples\_leaf(min number of data points allowed in a leaf node): we choose the space as [1, 2, 4].

The accuracy of the default random forest model is 0.40, and the average precision, recall and f1-score are 0.41. The micro-average area of PR curve is 0.36 and if ROC curve is 0.77.

For the random searched model, the best parameters we obtained are: n\_estimators: 900, min\_sample\_split: 2, min\_samples\_leaf: 1, max\_features: 'sqrt', max\_depth: 50. The accuracy is 0.583, the average precision, recall and f1-score are 0.59. According to the confusion matrices, among the 10 classes, the results are pretty close to each other, and the worst one is 'cat', the best one is 'snail'. The micro-average area of PR curve is 0.61 and if ROC curve is 0.90.

![Random Forest Confusion Matrix (Normalized)](images/rf_confusion_matrix_norm.png)
![Random Forest Confusion Matrix](images/rf_confusion_matrix_not_norm.png)

### Method 3: GBDT

A model using Gradient Boosted Decision Tree (GBDT) method is also adopted to recognize the images. This method is a combination of gradient descent and decision tree: it builds an additive model by adding a tree in each stage, with the tree computed by minimizing the loss function. In each update, a shrinkage parameter controls the learning speed of the boosting procedure, and subsampling a fraction of data in growing each tree adds to the independence between trees, thus improves the accuracy of the model.

We applied 4-fold cross validation here to select out the optimal model from our different-parameter combinations, where StratifiedKFold method is used to make sure the 4 divided sets from training set(80\% of the whole data set) contain roughly the same proportions of the 10 types of class labels.

**Parameters:**

In Gradient Boosted Decision Tree, two basic tuning parameters are the number of iterations and the sizes of each of the constituent trees, which in our function correspond to 'n\_estimators' and 'max\_depth'.

* Loss function: Referring to book The Elements of Statistical Learning, for Classification cases, the common loss function should be 'Deviance'.

* n\_estimators (iterations) are selected to be 100, 300, 500 or 1000. Even though Gradient boosting is fairly robust to over-fitting, we still used 100 as a candidate to reduce the probability of over-fitting. Also we set the iteration to stop while there is no change in continuous 30 times.

* subsample (The fraction of samples to be used for fitting the individual base learners): At the beginning, a set [0.4, 0.5, 0.6] is selected to identify the best case, where we recognized that the accuracy is generally increasing with higher subsample percentage and finally **1** is chosen to be the optimal option.

* max\_depth (maximum depth of the individual regression estimators): the parameter would control the number of nodes in the tree, whose best value depends on the interaction of the input variables. After preliminary trails, the results suggested us to try this with the range of [4~10].

* max\_features (the number of features to consider when looking for the best split): Here we choose ['sqrt', 'log2']. In our data set with 784(28 X 28) pixel features, their corresponding number of maximal features are 28 and 9.61. Therefore, intuitively with 'sqrt' the model would have a better performance, which is also verified by later outputs.

After grid search, under the condition where max iterations being 1000, by 4-fold cross validation, the training set being modelled with around 500 to 700 iterations, a final one is successfully searched out with a mean accuracy rate of the cross validation 0.6322.

The final optimal model is set with the following parameters:
learning\_rate is 0.05, max\_depth is 6, n\_estimators (iterations) is 1000, n\_iter\_no\_change is 30, max\_features is 'sqrt', max\_depth is **6**, learning\_rate is 0.05,  min\_samples\_split is 2 and min\_samples\_leaf is 1.

The final result demonstrates that after 630 iterations (the last 30 without any changes), the accuracy rate is around 63.717\%, micro-averaged average precision score over all classes is 0.63389, mean accuracy, micro-averaged precision score, recall score on the given test data and labels are 0.63717.

![GBDT ROC Curve](images/gbdt_ROC.png)
![GBDT PR Curve](images/images/gbdt_PR.png)

From the ROC and PR curves, the area under the micro-average ROC curve is 0.9035, the area under the micro-average PR curve is 0.63. According to the normalized confusion matrix, 'cat' is the class with worst prediction and 'snail' is the class with best prediction.


## Method 4: SVM
Supported vector machine(SVM), which is a supervised machine learning algorithm used for classification and we extend this method to multi-class classification here. It constructs set of hyperplanes in a high-dimensional space and uses kernel trick to transform our data. Based on these transformations it finds optimal boundaries between the possible outputs.  

We first do grid search with cross validation to find best parameter estimate of C and Gamma. Here we consider 4 kinds of kernel: linear, polynomial, gaussian and sigmoid. The overall results of 4 kernels are shown in the table below:

|Kernel | ACC | Precision | Recall | F1 | AUPR | AUROC|
|--- | --- | --- | --- | --- | --- | --- |
|linear | 0.45 | 0.45 | 0.45 | 0.44 | 0.39 | 0.82|
|poly | 0.65 | 0.65 | 0.65 |0.65 | 0.62 | 0.91|
|rbf | 0.60 | 0.61 | 0.60| 0.60 | 0.59 | 0.88|
|igmoid | 0.44 | 0.45 |0.44 | 0.44 | 0.38 | 0.81|

Among all kernels, we find polynomial kernel with parameter C = 10 and Gamma = 0.05 gives highest accuracy and average precision. Confusion matrix, ROC and PR curves also give best results as they return largest AUC. Hence we can conclude the choice of polynomial kernel with parameter C = 10 and Gamma = 0.05 as our final model in SVM and compare it with others methods proposed from other team members.

## Method 5: CNN
Convolutional Neural Network(CNN) is a class of deep, feed-forward ANN, most commonly applied to analyzing visual images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

Some strategies used to train CNN:

* Early-stoping: To avoid overfitting, we useearly-stopingwith validation set.

* ReLU:We use ReLU (rectified linearunit) as activation function.  ReLU functionis f(x) = max(0, x), where x is the input. It sets all negative values in the matrix X to 0 and keeps all the other values constant.  It isthe most used activation function since it re-duces training time and prevents the problem of vanishing gradients.

In this experiment, we explore the effects of different hyper-parameters and layer patterns to find the best model. The following elements are explored:

* Convolutional Layer Size: we compare 3X3 conv layers with 5X5 layers.

* Maxpooling  Layer: We try to train with or without Maxpooling Layer.

* Dropout Layer: We try train with or with-out Dropout Layer. And try different layer positions.

 – After Conv Layers:reduce over-fitting successfully, the performance on validation get better than without dropout.

 - After Dense Layers: reduce over-fitting better than dropout after conv layers.

 - After Both Conv & Dense: reduce over-fitting but not good as only dropout dense layers.


* Dense Layer: After multiple times of convolution, the outputs of convolutional layers represent the high-level semantics. To do classification, we can easily flatten the output from convolutional layers and connected to the output layers using softmax activation function. Instead of directly connected to the output layers, we try to add some fully connected layers before output layers, in which way it can learn nonlinear feature combinations easily and improve its performance.

* With Padding: We try train with or without zero Padding when using Conv Layers.

* Number of Convolutional Layers: Convolutional layers can do feature extraction. Convolutional layers extract the low-level features(pixels) to high-level, with more layers and deeper network, the model becomes more complex and thus can perform better, but also, are easily overfitting and hard to train(more parameters). We try 2, 4, 6, 8 Conv layers.

* Batch size: We try batch size 32,64, 128, 256, ..., etc.

We find our best model with ConvNet Architectures as follows produces validation accuracy = 0.8516.

![CNN Structure](images/cnnstructure.png)

The problem with small datasets is that models trained with them do not generalize well data from the validation and test set. Data augmentation is another way we can reduce over-fitting on models, where we increase the amount of training data using information only in our training data. We do rotation, zoom, width\_shift and height\_shift from our training data and tunning the range of them to find a best one which can augment effect data and not increase unnecessary noise. Finally, we choose rotation\_range = 10, zoom\_range = 0.1, width\_shift\_range = 0.1 and height\_shift\_range = 0.1. Using Data Augmentation, our best model improved greatly: the val\_acc is improved From 0.8516 to 0.8708.

The following figures show accuracy and loss from different selected CNN models.  

![CNN Model 1](images/CNN3.png)
![CNN Model 2](images/CNN4.png)

### Conclusion

In this project, we used KNN, SVM, GBDT, random forest and CNN to classify the quick draw dataset. The result shows that CNN achieves 86.25\% accuracy, beating other models with accuracy around 60\%. The classification for some classes has a high precision, such like for snail, which hits 99.45\%, whereas there are still some classes which have low precision such like dragon, since the impression of dragon in different people mind varies a lot.

**Predicted Accuracy for different Models**

|Model|Accuracy|
|------|------|
|KNN|0.605|
|Random Forest|0.583|
|GBDT|0.6372|
|SVM-Poly|0.65|
|CNN model|0.8625|

We can have a further look on the mis-classified image as shown below by our best model(CNN), we could find it is even impossible for human to recognize it. From this perspective, our model works well.

![CNN Final_Model](images/CNN6.png)

### Discussion

_Future work:_

* **More Complicated CNN-based model** More complicated models can be implemented, such as ResNet, MobileNet, etc.
* **Parallel Computation** Training on images takes a lot of computational resources and is really time-consuming. We can try to train models with GPUs or more CPUs to speed up our training process.
* **More features** we can try to import drawing-stroke information and timestamp, and use LSTM model, since maybe some images look the same but their drawing-stroke orders are different.

_Statement of Vision:_
Image recognition provides technical support for big data technologies such as automatic driving and face recognition, and is the cornerstone of building an intelligent society.
