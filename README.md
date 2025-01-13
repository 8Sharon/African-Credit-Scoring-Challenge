# African Credit Scoring Challenge - Predicting Loan Default Probability
This project aims to predict the probability of a customer defaulting on a loan based on available historical data. The goal is to create a machine learning model that classifies whether a customer will default on a loan or not. The dataset contains several customer attributes that may influence the likelihood of loan default. The solution leverages various machine learning techniques to preprocess the data, train models, and evaluate performance.
# Table of Contents
ID	
customer_id	
country_id	
tbl_loan_id	
lender_id	
loan_type	
Total_Amount	
Total_Amount_to_Repay	
disbursement_date	
due_date	
duration	
New_versus_Repeat
Amount_Funded_By_Lender	
Lender_portion_Funded
Lender_portion_to_be_repaid
target

# Dependencies
The following libraries and packages are required to run this project:

Python 3.7 or later
lightgbm: For the LGBMClassifier (Light Gradient Boosting Machine)
sklearn: For machine learning utilities like model evaluation, classification, and data preprocessing
pandas: For data manipulation and analysis
numpy: For numerical computations

# Data Preprocessing
The data contains both numerical and categorical features. It is important to preprocess the data to make it suitable for training a machine learning model.

# Steps:
Label Encoding: Convert categorical features to numerical values using LabelEncoder from scikit-learn.
Handling Imbalanced Classes: The dataset is likely to be imbalanced (more non-default than default), so class weights are computed using compute_class_weight to help the model handle this imbalance.

# Modeling
For this challenge, the LGBMClassifier (Light Gradient Boosting Machine) was chosen due to its high performance on structured/tabular data. 

# Steps:
Cross-validation: Utilize Stratified K-Fold (skfold) for cross-validation to ensure balanced class distribution in each fold.
Training the Model: Train the model using the optimized hyperparameters on the training set.

# Evaluation
Once the model is trained and optimized, its performance is evaluated using various metrics. The following metrics are used to assess the model's classification ability:

Accuracy: Proportion of correct predictions.
F1 Score: Harmonic mean of precision and recall, especially useful for imbalanced datasets.
Classification Report: A detailed report of precision, recall, f1-score, and support for each class (default and non-default).

# Results
The model's performance is evaluated using the F1 score, accuracy, and the classification report. The results will be displayed, showing how well the model performs in predicting loan defaults.
