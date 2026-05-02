# Heart Disease Classification and Model Evaluation

This project explores heart disease prediction using the UCI Heart Disease dataset. The main goal was to compare several machine learning methods for classifying whether a patient has heart disease based on clinical measurements. In addition to the main supervised learning models, the project also includes a PCA comparison and an exploratory hierarchical clustering section.

## Project Overview

The dataset contains 303 observations and 13 predictor variables describing patient health measurements such as age, chest pain type, cholesterol, blood pressure, exercise-induced angina, and other clinical indicators. The original target variable was converted into a binary outcome:

- `0` = no heart disease
- `1` = heart disease present

The following methods were used in the project:

- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)
- Hierarchical Clustering

## Main Results

The original model comparison showed that:

- **k-NN** achieved the strongest raw classification performance, with the highest accuracy, highest F1-score, and perfect recall on the test set.
- **Logistic Regression** achieved the highest ROC-AUC and was the strongest overall practical model because it combined strong predictive performance with interpretability.
- **SVM** also performed well, but did not outperform the other two main models.

The PCA extension showed that:

- Logistic Regression was nearly unchanged after PCA
- SVM improved slightly after PCA
- k-NN became noticeably worse after PCA

The hierarchical clustering extension showed that patients could be grouped into clusters with very different heart disease rates, suggesting meaningful structure in the data even without using the target label.

## Files in This Repository

- `isc4242_courseproject.pdf`  
  Final written report for the course project

- `heart_disease_project_notebook.ipynb`  
  Jupyter notebook containing all code for preprocessing, model fitting, evaluation, PCA, and hierarchical clustering

## Dataset Source

UCI Machine Learning Repository: Heart Disease Dataset

Reference:  
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository.  
DOI: https://doi.org/10.24432/C52P4X

## Methods Summary

### Preprocessing
- Converted the target variable into binary form
- Filled missing values in `ca` and `thal` using mode imputation
- Applied an 80/20 train-test split
- Standardized predictors using `StandardScaler`

### Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- ROC curves

### Model Tuning
- k-NN was tuned using 5-fold cross-validation over values of `k`
- SVM was tuned using 5-fold cross-validation over `C` and `gamma`
- PCA-based versions of k-NN and SVM were also tuned using 5-fold cross-validation

## Why This Project Matters

This project highlights an important point in machine learning: the model with the highest accuracy is not always automatically the best overall choice. In this case, k-NN had the strongest raw performance, but Logistic Regression provided a better balance of predictive strength, stability, and interpretability. Since this is a healthcare-related problem, model transparency and the cost of false negatives are important factors in deciding which method is most useful.

## Possible Future Improvements

Some possible extensions of this project include:

- testing more modern or larger heart disease datasets
- adding feature engineering
- comparing regularized classification models
- using repeated cross-validation for more stable performance estimates
- exploring additional interpretable machine learning methods

## Author

Jonathan Snyder
