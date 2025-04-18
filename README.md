# African Credit Scoring Challenge - Predicting Loan Default Probability

The African Credit Scoring Challenge focuses on predicting the probability of loan default based on customer and loan-related features. The goal is to develop a machine learning model that can predict whether a loan will default or not, using historical data of borrowers and their loan details. 

This challenge involves preprocessing a dataset containing both numerical and categorical features such as loan amount, loan type, customer status, and repayment history. The model is evaluated on its ability to correctly predict loan defaults, which is crucial for lenders to assess the risk of granting loans to potential customers.

By addressing the imbalance in the target variable (default vs. non-default), the project aims to create a robust model for predicting loan defaults in African markets, where credit scoring data is often scarce or incomplete.



# ⚠️ Disclaimer
All datasets, information, and reports within this repository are fictional and created solely for illustrative purposes to showcase advanced predictive machine learning techniques. They do not include any real proprietary, confidential, or sensitive information related to any company, organization, or individual.

![credit](https://media.istockphoto.com/id/1142099845/photo/businessman-giving-money-south-korean-won-bills-to-his-partner-at-the-desk.jpg?s=612x612&w=0&k=20&c=L9v62yJRRQSZvDVOiwRLqKafNsamYM3FJ34GLMb42e0=)

## Table of Contents
1. [Objective](#objective)
2. [Problem Statement](#problem-statement)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [How to Run the Code](#how-to-run-the-code)
9. [Contact Information](#contact-information)

## Objective
The objective of this project is to predict the probability of a customer defaulting on a loan based on their historical data. By training a machine learning model, this project aims to classify whether a customer will default on their loan or not, helping lenders make informed decisions on granting loans. The model will be built using various techniques for preprocessing, model training, and evaluation.

## Problem Statement
In many regions, particularly in Africa, loan defaults can have significant financial implications for both the borrower and the lender. The challenge is to predict the likelihood that a customer will default on a loan based on historical data. This can help lenders assess the creditworthiness of potential borrowers and make more informed lending decisions.

The dataset used in this project includes various features like customer information, loan details, and repayment information. By analyzing these features, we can predict whether a customer will default on a loan (target variable). The solution involves applying machine learning algorithms, handling data preprocessing tasks, and optimizing the model to improve predictive performance.

## Dependencies
To run this project, the following libraries and packages are required:

- **Python 3.7 or later**
- **lightgbm**: For implementing the LGBMClassifier (Light Gradient Boosting Machine)
- **sklearn**: For model evaluation, classification, and data preprocessing utilities
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations

# Loan Default Prediction

This repository contains the implementation of a machine learning model to predict loan defaults based on various loan features. The model is built using the Light Gradient Boosting Machine (LGBMClassifier) due to its strong performance with structured/tabular data and its ability to handle categorical features directly.

## Data Preprocessing

The dataset includes both numerical and categorical features. To prepare the data for training, the following preprocessing steps were applied:

### 1. Label Encoding
Categorical features were converted into numerical values using the `LabelEncoder` from `sklearn`. This step ensures that categorical variables can be used effectively by machine learning algorithms.

### 2. Handling Imbalanced Classes
The target variable, `target`, is imbalanced (more non-defaults than defaults). To address this, the `compute_class_weight` function was used to compute the class weights. These weights are used during model training to help the model handle the imbalance and avoid bias towards the majority class.

### Key Features in the Dataset
- **customer_id**: Unique identifier for the customer
- **country_id**: The country where the loan was issued
- **tbl_loan_id**: Loan identifier
- **lender_id**: Lender providing the loan
- **loan_type**: Type of loan (e.g., personal, business)
- **Total_Amount**: Total loan amount
- **Total_Amount_to_Repay**: Total amount to be repaid (including interest)
- **disbursement_date**: Date the loan was issued
- **due_date**: Date the loan is due for repayment
- **duration**: Duration of the loan in months
- **New_versus_Repeat**: Whether the borrower is a new or repeat customer
- **Amount_Funded_By_Lender**: Amount funded by the lender
- **Lender_portion_Funded**: Portion of the loan funded by the lender
- **Lender_portion_to_be_repaid**: Portion of the lender’s funded amount to be repaid
- **target**: The target variable indicating whether the loan was defaulted (1) or not (0)

## Modeling

The model to predict loan defaults was built using the **Light Gradient Boosting Machine (LGBMClassifier)**. This classifier was chosen for its high performance on structured/tabular data and its ability to handle categorical features directly.

### Steps in the Modeling Process:
1. **Cross-validation**:
   - Stratified K-Fold cross-validation was used to ensure balanced class distribution in each fold. This is crucial for imbalanced datasets, as it prevents the model from being trained only on the majority class.

2. **Model Training**:
   - The LGBMClassifier was trained on the training dataset with optimized hyperparameters. Class weights, computed earlier, were incorporated to account for the imbalanced classes.

3. **Hyperparameter Tuning**:
   - Hyperparameters of the model were fine-tuned using techniques like grid search to optimize the model’s performance.

## Evaluation

The model’s performance was evaluated using the following metrics:
- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall, which is particularly useful for imbalanced datasets where accuracy alone can be misleading.
- **Classification Report**: Detailed report showing precision, recall, F1-score, and support for each class (default and non-default).

The results provide insight into how well the model predicted loan defaults and indicate whether it was more inclined towards predicting defaults or non-defaults.

## Results
The model's performance metrics (Accuracy, F1 Score, and Classification Report) were evaluated to determine how well the model could predict loan defaults. The results provide an understanding of the model's prediction capabilities and biases

# 📬 Contacts
For questions or feedback, feel free to open an issue or contact the repository owner.

- **Author**: Sharon Kamau
- **Email**: [njerisharon611@gmail.com](njerisharon611@gmail.com)
