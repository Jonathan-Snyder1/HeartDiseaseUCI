# HeartDiseaseUCI
https://archive.ics.uci.edu/dataset/45/heart%2Bdisease

Intro/
The objective of this project is to build and evaluate machine learning models for predicting the presence of heart disease using the UCI Heart Disease dataset. This dataset is well suited for a classification task because it contains 303 patient records, 13 commonly used clinical features, and a target variable indicating the presence or absence of heart disease. The dataset is especially appropriate for this project because the Cleveland subset has been widely used in prior machine learning studies, and the target is commonly simplified into a binary outcome by treating values 1–4 as disease present and 0 as no disease. In this project, I will compare several classification methods, including Logistic Regression, k-Nearest Neighbors, and Support Vector Machines, to determine which model provides the strongest predictive performance. I will also examine preprocessing steps such as handling missing values, scaling features, and possibly applying dimensionality reduction with Principal Component Analysis. Model performance will be assessed using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Beyond model accuracy, this project will also consider the practical importance of interpretability, as heart disease prediction is a healthcare problem where both predictive strength and transparency matter.

This project focuses on building classification models to predict whether a patient has heart disease using the UCI Heart Disease dataset. The dataset is a strong choice for this assignment because it contains real clinical variables such as age, chest pain type, cholesterol, blood pressure, and maximum heart rate, along with a target variable that indicates the presence of heart disease. The UCI description notes that the Cleveland version of the dataset is the one most commonly used in machine learning research, and that the target is often converted into a binary classification problem by grouping values 1 through 4 as heart disease present and 0 as no disease. For this project, I plan to compare Logistic Regression, k-Nearest Neighbors, and Support Vector Machines in order to see which method performs best on this medical classification task. I will also apply preprocessing steps such as dealing with missing values and scaling the predictors where needed. Model performance will be evaluated using classification metrics like accuracy, precision, recall, F1-score, and ROC-AUC, and the final discussion will consider both model performance and the real-world importance of reliable and interpretable predictions in healthcare settings.



Title

Heart Disease Classification Using Logistic Regression, k-Nearest Neighbors, and Support Vector Machines

1. Introduction
Background and motivation
Project objective
Why this is a classification problem
Preview of models and evaluation metrics
2. Dataset Description
Source of data
Number of observations and variables
Description of predictors and target
Why the dataset was chosen
3. Data Preprocessing
Data cleaning and missing values
Binary target construction
Feature preparation and scaling
Train/test split
Exploratory analysis
4. Methodology
4.1 Logistic Regression
4.2 k-Nearest Neighbors
4.3 Support Vector Machine
4.4 Hyperparameter Tuning
5. Model Performance Assessment
Evaluation metrics
Results comparison table
Confusion matrices
ROC curves
Best model selection
6. PCA Extension
Purpose of PCA
Results after dimensionality reduction
Comparison with original models
7. Real-World Interpretation and Discussion
Important predictors
Practical applications
Limitations
Ethical considerations
8. Conclusion
Summary of findings
Best-performing model
Interpretability vs performance






# 1. Introduction

Heart disease remains one of the most important health problems worldwide, making early detection and risk assessment valuable areas for data-driven modeling. In many healthcare settings, patient information such as age, blood pressure, cholesterol level, chest pain type, and exercise-related measurements can provide useful clues about the likelihood of heart disease. Because of this, heart disease prediction is a strong example of a real-world classification problem where machine learning methods can be used to support decision-making.

The objective of this project is to build and compare classification models that predict the presence of heart disease using patient clinical measurements from the UCI Heart Disease dataset. This dataset is especially well suited for the project because it is designed for classification tasks and has been widely used in machine learning research. The target variable represents the presence of heart disease, and the dataset contains a set of medically relevant predictors that can be used to estimate that outcome. On the UCI dataset page, the Cleveland database is identified as the version most commonly used in machine learning experiments, and prior work often simplifies the target into a binary outcome by distinguishing disease presence, coded as values 1 through 4, from disease absence, coded as 0.

In this project, three classification methods will be compared: Logistic Regression, k-Nearest Neighbors, and Support Vector Machines. These models were chosen because they represent different approaches to classification and also connect closely to topics covered in this course, including model evaluation, distance-based learning, and margin-based classification. Logistic Regression provides a strong and interpretable baseline for binary classification, k-Nearest Neighbors offers a flexible nonparametric method based on local similarity, and Support Vector Machines provide a more advanced approach that can capture more complex decision boundaries. Model performance will be assessed using several common classification metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. The final discussion will consider not only predictive performance, but also the importance of interpretability and practical limitations in a healthcare setting. This overall structure aligns closely with the project requirement to justify the data and model choice, evaluate model performance carefully, and interpret the results in a meaningful real-world context.

# 2. Dataset Description

The dataset used in this project is the Heart Disease dataset from the UCI Machine Learning Repository. According to the UCI documentation, the full database contains 76 attributes collected from four sources: Cleveland, Hungary, Switzerland, and the VA Long Beach database. However, most published experiments use a subset of 14 attributes, and the Cleveland database is the version most commonly used by machine learning researchers. The dataset is categorized by UCI as a multivariate classification dataset in the Health and Medicine subject area. It contains 303 instances, includes 13 predictive features plus the target variable, and contains a mix of categorical, integer, and real-valued variables. The UCI page also notes that the dataset contains missing values.

The response variable in this dataset is the field called goal or num, which represents the diagnosis of heart disease. It is originally coded from 0 to 4, where 0 indicates no presence of heart disease and higher values indicate disease presence. A common approach in prior experiments is to convert this into a binary target by grouping values 1, 2, 3, and 4 together as “heart disease present” and keeping 0 as “no heart disease.” This binary setup is appropriate for the current project because it creates a clear classification task and allows for direct comparison of different supervised learning methods.

The predictor variables include several clinically meaningful measurements, such as age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise induced angina (exang), ST depression induced by exercise relative to rest (oldpeak), slope of the peak exercise ST segment (slope), number of major vessels colored by fluoroscopy (ca), and thal. These variables make the dataset well suited for model building because they combine demographic, physiological, and exercise-related information that may all be relevant for predicting disease status.

This dataset was chosen for several reasons. First, it directly supports the type of classification task required for the course project. Second, the medical context gives the analysis real-world relevance beyond simply producing a prediction score. Third, the dataset is manageable in size, making it practical for comparing multiple models and preprocessing strategies within the scope of this assignment. Finally, the dataset fits especially well with course topics such as classification, model comparison, preprocessing, and performance evaluation.


# 3. Data Preprocessing

Before building the classification models, several preprocessing steps were necessary to prepare the Heart Disease dataset for analysis. The UCI documentation indicates that the dataset contains missing values, includes a mix of categorical, integer, and real-valued features, and is commonly analyzed using the Cleveland processed version with 13 predictors and one target variable. Because machine learning models depend heavily on the quality and format of the input data, preprocessing plays an important role in making the comparison between models both fair and meaningful.

The first preprocessing step was to inspect the dataset for missing values and confirm the structure of the variables. Since the target variable is originally coded from 0 to 4, where 0 indicates no heart disease and values 1 through 4 indicate disease presence, the response variable was converted into a binary outcome for this project. Specifically, observations with a target value of 0 were labeled as 0 = no heart disease, while observations with target values of 1, 2, 3, or 4 were labeled as 1 = heart disease present. This binary transformation is consistent with the standard experimental setup described by UCI and makes the dataset appropriate for direct comparison of binary classification methods.

After constructing the target variable, the predictor variables were reviewed for their measurement types and practical modeling use. The dataset contains several predictors that are numeric by nature, such as age, resting blood pressure, cholesterol, maximum heart rate achieved, and oldpeak, along with several variables that are categorical or category-coded, such as sex, chest pain type, fasting blood sugar, resting ECG results, exercise-induced angina, slope, ca, and thal. In the preprocessing stage, these features were kept as model inputs, while making sure their values were correctly represented and usable for classification. Depending on the final coding workflow, some category-coded variables may be left in their original encoded form or treated more explicitly during analysis, but all were retained because they contain clinically relevant information tied to heart disease diagnosis.

The data was then divided into training and testing sets so that model performance could be evaluated on unseen observations rather than only on the data used for fitting. This is an important part of predictive modeling because training performance alone can be misleadingly optimistic, a point that is emphasized in the course review guide when distinguishing the roles of training, validation, and test data. A train/test split allows the final reported metrics to better reflect how well the models generalize to new cases.

Feature scaling was also included as part of preprocessing, particularly for k-Nearest Neighbors and Support Vector Machines. These models are sensitive to the scale of the predictors because they rely on distances or geometric boundaries in feature space. Variables such as cholesterol, resting blood pressure, and maximum heart rate are measured on different numerical scales, so standardizing the predictors helps prevent variables with larger magnitudes from dominating the model. Logistic Regression can also benefit from scaling when regularization is involved, but scaling is especially important for k-NN and SVM in order to produce a fair model comparison.

In addition to these preparation steps, exploratory checks such as examining the class distribution, summary statistics, and feature relationships may also be performed before modeling. These checks can help identify any imbalance between the heart disease and non-heart disease classes, unusually distributed predictors, or potential data quality concerns. Overall, the preprocessing stage ensures that the dataset is cleaned, consistently formatted, and ready for model fitting and evaluation. This is especially important in a healthcare-focused classification problem, where both model performance and careful data handling matter.

# 4. Methodology

To evaluate heart disease prediction performance, this project compares three supervised classification models: Logistic Regression, k-Nearest Neighbors (k-NN), and Support Vector Machines (SVM). These models were selected because they represent different approaches to classification and provide a useful basis for comparison on a medical dataset. Logistic Regression serves as a strong baseline model for binary classification and offers interpretability, k-NN provides a flexible distance-based method that classifies observations according to their nearby neighbors, and SVM offers a margin-based approach that can capture more complex class boundaries. Using multiple methods makes it possible to compare not only predictive performance, but also tradeoffs between flexibility, interpretability, and computational complexity. This type of comparison is consistent with the example report structure and with the course emphasis on comparing different modeling approaches rather than relying on a single method.

# 4.1 Logistic Regression

The first model used in this project is Logistic Regression, a standard method for binary classification problems. Logistic Regression estimates the probability that an observation belongs to a given class by modeling the relationship between the predictors and the log-odds of the outcome. In this case, the model estimates the probability that a patient has heart disease based on the clinical predictors in the dataset. Because the output is expressed as a probability between 0 and 1, Logistic Regression is especially useful when the goal is to classify observations into two groups, such as heart disease present versus heart disease absent.

One reason for including Logistic Regression is that it provides a strong and interpretable baseline. In healthcare-related applications, interpretability is important because the model coefficients can help show how changes in predictors are associated with increased or decreased odds of disease. Logistic Regression also connects directly to the course theme of building and evaluating predictive models while balancing model simplicity and performance. In this project, Logistic Regression will be trained on the processed dataset and evaluated using the same performance metrics as the other models so that the comparison remains fair.

# 4.2 k-Nearest Neighbors

The second model used is k-Nearest Neighbors (k-NN). Unlike Logistic Regression, k-NN is a nonparametric method that does not fit a single global equation to the data. Instead, it classifies a new observation by examining the labels of the k closest training observations in feature space and assigning the majority class among those neighbors. This makes k-NN a local, distance-based method whose predictions depend heavily on the surrounding data points.

k-NN was chosen because it provides a useful contrast to Logistic Regression. While Logistic Regression assumes a particular functional relationship between predictors and the log-odds of the outcome, k-NN makes fewer structural assumptions and can adapt to more irregular decision boundaries. However, its performance depends strongly on the choice of k, and it is sensitive to the scale of the predictors. For that reason, feature standardization is an important preprocessing step before fitting the k-NN model. In this project, the value of k will be selected through model tuning so that the classifier is not based on an arbitrary choice. This model also fits closely with the class material, since the review guide emphasizes how k-NN uses local geometry and how different values of k affect flexibility and bias-variance tradeoffs.

# 4.3 Support Vector Machine

The third model used is the Support Vector Machine (SVM). SVM is a supervised learning method that constructs a decision boundary designed to maximize the margin between classes. In a binary classification setting, the model attempts to separate observations with heart disease from those without heart disease by finding the hyperplane that best distinguishes the two groups. The observations closest to this boundary are called support vectors, and they play the most important role in determining the model.

SVM was included because it provides a more advanced classification approach that can perform well when the class boundary is not captured as effectively by simpler methods. Depending on the implementation, SVM can be used with either a linear boundary or a nonlinear kernel, allowing it to adapt to more complex patterns in the data. Because SVM is based on distances and geometry in feature space, predictor scaling is also important before fitting this model. In this project, SVM will be compared directly with Logistic Regression and k-NN to determine whether the additional flexibility leads to better classification performance on the heart disease dataset. The course review guide also highlights SVM as a major classification topic, especially in terms of margin-based reasoning and geometric interpretation.

# 4.4 Hyperparameter Tuning

To improve model performance and ensure a fair comparison, hyperparameter tuning will be used for the models where tuning is appropriate. For Logistic Regression, tuning may involve adjusting the regularization strength to control model complexity. For k-NN, the most important hyperparameter is the number of neighbors, k, since smaller values can lead to highly flexible decision boundaries while larger values produce smoother classifications. For SVM, hyperparameter tuning may include the penalty parameter and, if a nonlinear kernel is used, kernel-specific settings that affect the shape of the decision boundary.

These tuning choices will be made using validation procedures such as cross-validation on the training data. This approach is important because it reduces the risk of selecting model settings based only on a single random split and better reflects the course emphasis on separating training, validation, and testing roles in model evaluation. After tuning, the final selected version of each model will be evaluated on the held-out test set using the same performance metrics.

# 4.5 Evaluation Strategy

Once the models are trained, they will be compared using several classification metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. These measures provide a broader picture of model quality than accuracy alone. Accuracy summarizes overall correctness, while precision and recall help show how well the models handle positive heart disease cases. The F1-score balances precision and recall, and ROC-AUC helps evaluate how well the classifier separates the two classes across different thresholds. Confusion matrices will also be used to visualize the types of classification errors made by each model. This is especially relevant in a healthcare context, where false negatives and false positives may have different real-world consequences.

# 4.6 Optional PCA Extension

As an additional extension, Principal Component Analysis (PCA) may be used after the initial model comparison. PCA is a dimensionality reduction technique that transforms the original predictors into a smaller set of orthogonal components that capture the main sources of variation in the data. In this project, PCA can be used to examine whether reducing dimensionality affects predictive performance and whether the data becomes easier to visualize in a lower-dimensional space. This extension connects to another major topic in the course and allows the project to compare performance both before and after dimensionality reduction.

# 5. Model Performance Assessment

To compare the effectiveness of the classification models, their predictive performance was evaluated using several standard metrics: accuracy, precision, recall, F1-score, and ROC-AUC. These metrics were chosen because they provide a broader view of performance than accuracy alone. In a healthcare classification problem such as heart disease prediction, it is important not only to measure overall correctness, but also to consider how well the model identifies patients who actually have heart disease. This aligns with the project requirement to assess model performance using appropriate evaluation criteria and relevant visualizations.

# 5.1 Evaluation Metrics

Each model was evaluated on the held-out test set using the same group of classification metrics. Accuracy measures the proportion of correctly classified observations out of all cases. While this is a useful overall summary, it does not always capture the full picture, especially if the two classes are not perfectly balanced. Precision measures the proportion of predicted positive cases that were actually positive, which is useful when false positives are costly. Recall, also called sensitivity, measures the proportion of actual positive cases that were correctly identified by the model. In the context of heart disease prediction, recall is especially important because failing to identify a patient who truly has heart disease may be more serious than incorrectly flagging a healthy patient. F1-score combines precision and recall into a single measure, making it useful when both types of error matter. Finally, ROC-AUC evaluates how well a model separates the two classes across different classification thresholds, with higher values indicating stronger discrimination ability.

# 5.2 Comparative Model Results

After preprocessing and training, the three models — Logistic Regression, k-Nearest Neighbors, and Support Vector Machine — were compared based on their test-set performance. The results can be summarized in a comparison table showing the values of accuracy, precision, recall, F1-score, and ROC-AUC for each method. Presenting the results in a single table makes it easier to identify which model achieved the strongest overall performance and which models may have performed better on specific metrics.

You can drop in a table like this later:

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	XX	XX	XX	XX	XX
k-Nearest Neighbors	XX	XX	XX	XX	XX
Support Vector Machine	XX	XX	XX	XX	XX

This comparison is important because different models may excel in different areas. For example, one model may achieve the highest overall accuracy, while another may produce better recall for positive heart disease cases. Since the project is not only about maximizing one number, the final model choice should consider performance across multiple criteria rather than relying on accuracy alone.

# 5.3 Confusion Matrices

In addition to summary metrics, confusion matrices were used to examine the classification behavior of each model in more detail. A confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. This is especially useful in a medical context because it helps show the kinds of mistakes the model is making. For example, a false negative corresponds to a patient with heart disease being classified as healthy, which may be more concerning than a false positive in a screening-related setting.

By comparing confusion matrices across the three models, it becomes possible to see whether one model tends to miss more positive cases or whether another model incorrectly labels too many healthy patients as high-risk. This deeper view helps support the interpretation of the precision and recall values and gives a more complete understanding of each classifier’s strengths and weaknesses.

# 5.4 ROC Curves and AUC

To further compare the models, Receiver Operating Characteristic (ROC) curves were used along with the Area Under the Curve (AUC) statistic. ROC curves show the relationship between the true positive rate and false positive rate across a range of classification thresholds. A model with a curve closer to the upper-left corner of the graph generally has stronger classification performance, while a model with a larger AUC value has better ability to distinguish between patients with and without heart disease.

Including ROC curves adds an important visual comparison to the project, since two models with similar accuracy may still differ in how effectively they separate the two classes overall. This type of performance visualization also follows the style of the example classification report, which included ROC-based comparisons in addition to standard metrics.

# 5.5 Best Model Selection

Based on the evaluation metrics and visualizations, the best-performing model will be selected by considering both predictive strength and practical usefulness. If one model achieves the highest values across most metrics, then it would be the strongest candidate overall. However, if another model performs similarly while offering greater interpretability, that tradeoff should also be considered. For example, Logistic Regression may be easier to explain in a healthcare setting, while SVM or k-NN may provide slightly different predictive advantages depending on the structure of the data.

The final model choice should therefore be based on a balanced interpretation of the results. This approach reflects an important course idea: model evaluation is not only about fitting a model, but also about comparing alternatives and justifying why one method is preferable in context.

# 6. PCA Extension

As an additional extension to the main classification analysis, Principal Component Analysis (PCA) was considered as a dimensionality reduction technique. PCA transforms the original predictors into a smaller set of new variables called principal components, where each component is a linear combination of the original features and captures a portion of the variation in the data. The first principal component explains the greatest amount of variance, while each later component explains the next largest amount subject to being orthogonal to the previous components. This connects directly to a major topic from the course, where PCA is presented as a way to reduce dimensionality, visualize data more easily, and sometimes improve modeling efficiency.

In this project, PCA can be used after the original preprocessing steps and before fitting the classification models. The main reason for including PCA is not necessarily to outperform the original feature set, but rather to explore whether a lower-dimensional representation of the data still preserves enough information for effective heart disease classification. If the models perform similarly after dimensionality reduction, then PCA may offer the benefit of simplifying the feature space and making the data easier to visualize. If performance drops, that would suggest that the original clinical variables contain important information that is lost when the predictors are compressed into fewer components.

A useful way to implement this extension is to standardize the predictors first, apply PCA, and then select a reduced number of principal components based on explained variance. For example, the analysis may compare model performance using the original predictors versus a PCA-transformed version of the dataset. In addition, a two-dimensional PCA scatterplot can be included to visualize the observations along the first two principal components. While PCA does not use class labels and therefore does not explicitly optimize class separation, it may still reveal whether patients with and without heart disease appear somewhat distinguishable in a reduced feature space. This also creates a natural contrast with supervised methods such as Logistic Regression, k-NN, and SVM, which are designed directly for classification.


If PCA is included in the final results, the discussion should focus on whether dimensionality reduction improved interpretability, visualization, or computational simplicity, and whether any changes in predictive performance were small or substantial. This style of extension is similar to the example classification report you uploaded, where PCA was used as a secondary analysis to compare model performance before and after dimensionality reduction rather than replacing the main classification workflow. Overall, PCA adds a useful course-related component to the project and helps show a broader understanding of model-building strategies beyond fitting a single classifier.


# 7. Real-World Interpretation and Discussion

Although model performance metrics are important, they do not fully capture the practical meaning of the results. In a healthcare-related classification problem, it is also necessary to interpret what the model is learning, how its predictions might be used, and what limitations or ethical concerns should be acknowledged. The course project instructions specifically ask for interpretation of key patterns, discussion of limitations or bias, and suggestions for future improvement, so this section is essential for connecting the technical results to a real-world setting.

One important part of the discussion is identifying which variables appear to be most influential in predicting heart disease. Depending on the final fitted models, variables such as chest pain type, maximum heart rate achieved, exercise-induced angina, oldpeak, or number of major vessels may emerge as especially informative predictors. If Logistic Regression performs well, its coefficients can help show the direction and relative importance of these predictors, making the model easier to interpret than more opaque alternatives. This interpretability can be valuable in a medical setting, where understanding why a prediction was made may be almost as important as the prediction itself.

At the same time, predictive performance should be interpreted carefully. A model with strong accuracy or ROC-AUC may still make errors that matter clinically. In this project, false negatives are especially important because they represent patients who actually have heart disease but are predicted not to have it. In a real screening or triage context, this type of error could delay follow-up testing or treatment. False positives also matter, since they could lead to unnecessary anxiety or additional procedures, but in many healthcare settings the cost of missing a real case may be more serious. For this reason, the final interpretation should pay close attention to recall and the confusion matrix, not just overall accuracy.

There are also several limitations that should be acknowledged. First, the dataset is relatively small, which means the final model performance may be sensitive to the specific train/test split and may not generalize perfectly to broader patient populations. Second, the UCI Heart Disease data is older and may not fully reflect current clinical practice, population health patterns, or modern measurement standards. Third, converting the target variable into a binary outcome simplifies the problem and may hide differences in severity among patients with different levels of disease. Finally, missing values and dataset-specific preprocessing decisions can also affect the results, especially when comparing multiple models. These kinds of limitations are important to state clearly because they show that the analysis is being interpreted responsibly rather than treated as universally valid.

Ethical considerations also matter in this project. A predictive model for heart disease should be viewed as a decision-support tool, not as a replacement for medical judgment. Even if the model performs well, it should not be used in isolation to diagnose or treat patients. Bias in the dataset, limited representation of certain patient groups, or class-specific errors could all affect fairness and reliability. In addition, a highly accurate model that is difficult to interpret may be less useful in practice than a slightly simpler model that clinicians can better understand and trust. This creates an important tradeoff between raw predictive performance and interpretability, especially in medical applications where accountability and explanation are important.

Overall, this project demonstrates how machine learning methods can be used to approach a medically meaningful classification problem. By comparing different models and examining both their strengths and limitations, the analysis goes beyond simply reporting which classifier performed best. Instead, it shows how model choice depends not only on accuracy, but also on interpretability, error type, and practical context. This broader interpretation makes the project more aligned with real-world data science, where the best model is not always just the one with the highest score.


# 8. Conclusion

The goal of this project is to build and compare machine learning models for predicting the presence of heart disease using clinical measurements from the UCI Heart Disease dataset. By framing the problem as a binary classification task, the analysis focuses on determining how effectively different supervised learning methods can distinguish between patients with and without heart disease. This project is well aligned with the objectives of the course because it combines data preprocessing, model fitting, model comparison, performance evaluation, and interpretation in a single applied setting.

Three classification models were selected for comparison: Logistic Regression, k-Nearest Neighbors, and Support Vector Machines. These models represent different strategies for classification and provide a useful balance between interpretability and predictive flexibility. Logistic Regression offers a simple and interpretable baseline, k-NN provides a local distance-based approach, and SVM introduces a margin-based method that may capture more complex class boundaries. Their performance will be compared using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, along with confusion matrices and ROC curves. This comparison makes it possible to identify not only which model performs best numerically, but also which model is most practical in a healthcare context.

An additional PCA extension was included to explore whether dimensionality reduction could preserve useful predictive structure while simplifying the feature space. Even if PCA does not improve classification performance, it still provides value by connecting the project to another major course topic and by offering a clearer way to visualize the data in lower dimensions. This makes the project more comprehensive while keeping the main focus on classification and model evaluation.

Overall, this project is designed to show that model building is more than simply fitting an algorithm and reporting its accuracy. A complete analysis also requires thoughtful preprocessing, careful performance assessment, and consideration of real-world interpretation, limitations, and ethical concerns. In the context of heart disease prediction, this broader approach is especially important because predictive models may one day support high-impact decisions in healthcare environments. Future improvements could include testing additional models, using larger or more modern medical datasets, and exploring more detailed feature engineering or validation strategies. By comparing multiple methods and interpreting their results carefully, this project aims to provide a meaningful example of applied classification in data science.

