# Machine Learning | Risk Report

## Overview of the Analysis


The purpose of this analysis is to identify the creditworthiness of borrowers using a dataset of historical lending activity from a peer-to-peer lending services company to train and evaluate models with imbalanced classes.

The historical lending activity used for this analysis is stored in the csv file **lending_data.csv** which contains the following variables:
* Dependent (y) Label set: loan status
* Independent (x) Features set: loan size, interest rate, borrower income, debt to income, num of accounts, derogatory marks, total debt 


In order to check the balance of the label set **value counts** was used. This function counts each category: the healthy loans (0) and the high-risk loans (1).

Using the train_test_split function from from sklearn.model_selection the data was split into training and testing datasets
```
training_x, testing_x, training_y, testing_y = train_test_split(x,y)
```
This functions generates training data for features (x) and label (y) sets: training_features, testing_features, training_labels, testing_labels. These variables are useful for the next steps in the analysis which is creating a predictive model following the pattern model-fit-predict:

* **Model:** Create a Logistic Regression Model (from sklearn.linear_model)
```
model  = LogisticRegression(random_state=1)
```
* **Fit:** Training stage using the model. To fit, or train, the model to the training data, we can use the fit function that scikit-learn makes available to the logistic regression classifier, as the following code shows:
```
model.fit(training_features,training_labels)
```
* **Predict:** Using the trained model to predict new data. The predict function classifies the features and discovers if the model can assign them to the correct targets.
```
predictions = model.predict(testing_features)
```
* **Evaluate the model:**  Using the model evaluation technique called the accuracy_score() function
```
balanced_accuracy_score(testing_labels, predictions)
```
 * **Confusion Matrix**: A confusion matrix (`confusion_matrix`) groups our model’s predictions according to whether the model accurately predicted the categories (True Positives, False negatives, False positives, True Negatives)
```
matrix = confusion_matrix(testing_labels, predictions)
```
* **Classification Report**: Calculate the metrics for our model (e.g., `precision, recall`)
```
report = classification_report_imbalanced(testing_labels,predictions)
```
As part of this analysis, a second Model was created using the `RandomOverSampler` function to resample the data. This randomly select instances of the minority class and add them to the training set until the majority and minority classes are balanced
```
random_oversampler = RandomOverSampler(random_state=1)
X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
```

## Results


* Machine Learning Model 1:
  * Uses the original training data
  * The precision and the recall for 0 (healthy loan) are better than that for 1 (high-risk loan). The precision for the 0 values is 1 which means that out of all the times that the model predicted a testing data to be the value 0, 100% of those predictions were correct. On the other hand, out of all the times that the model predicted a value of 1, only 85% of those predictions were correct.

  _

  ```
                     pre       rec       spe        f1       geo       iba       sup

          0         1.00      0.99      0.91      1.00      0.95      0.91     18765
          1         0.85      0.91      0.99      0.88      0.95      0.90       619
        
        avg/total   0.99      0.99      0.91      0.99      0.95      0.91     19384

  ```



* Machine Learning Model 2:
  * This model uses the `RandomOverSampler` module from the imbalanced-learn library to resample the data.
  * The precision and the recall for 0 (healthy loan) are better than that for 1 (high-risk loan). The precision for the 0 values is 1 which means that out of all the times that the model predicted a testing data to be the value 0, 100% of those predictions were correct. On the other hand, out of all the times that the model predicted a value of 1, only 84% of those predictions were correct.
  
  _

  ```
                     pre       rec       spe        f1       geo       iba       sup

          0         1.00      0.99      0.99      1.00      0.99      0.99     18765
          1         0.84      0.99      0.99      0.91      0.99      0.99       619
        
        avg/total   0.99      0.99      0.99      0.99      0.99      0.99     19384

  ```

## Summary

After training the models with unbalanced and unbalance data, and getting different results in terms of precision and recall, we can see that the precision of the second model (resampled balanced using `RandomOverSampler`) is 1% less accurate than the previous model that uses the original data, when predicting the high risk loans.

It becomes obvious that it is harder and more important to have a higher accuracy when predicting high risk loans and since the model that uses the original unbalanced data seems to perform better (precision 1% better) I recommend using the first Model with the original data.

---

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


