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

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
