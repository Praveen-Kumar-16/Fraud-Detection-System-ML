# Fraud Detection for Transaction Systems

## Overview:
This project demonstrates a fraud detection mechanism for transaction systems using machine learning algorithms. The objective is to classify transactions as either suspicious or legitimate to enhance financial security. The project is contained within a single Jupyter Notebook file, main.ipynb, which includes code for data preparation, model training, evaluation, and testing.


## Features:
  
    Data Preparation: Load and preprocess transaction data for model training.
    Model Building: Train and evaluate Decision Tree and Random Forest classifiers.
    Model Saving: Save trained models for future predictions.
    Model Testing: Apply models to new data for fraud detection.
    Cross-Validation: Validate model performance with test data.


## Technologies Used:
 
    Python: Programming language for all the code and analysis. 
    Pandas: Library for data manipulation and analysis.
    Scikit-Learn: Library for building machine learning models.
    Joblib: Library for saving and loading machine learning models.
    Jupyter Notebook: Interactive environment for developing and documenting code.

## Getting Started:

### To get started with this project, follow these steps:

  #### Clone the Repository:
  
     git clone https://github.com/yourusername/fraud-detection-project.git
     cd fraud-detection-project

#### Install Required Packages

Ensure you have Python 3.x installed. Then install the necessary packages using pip:

    pip install pandas scikit-learn joblib

## Add Your Data Files

### Place your data files in the data directory:

    data/transactions.xlsx: Contains the transaction data used for model training.
    data/test_data.xlsx: Contains new transaction data for model testing.

## Open the Jupyter Notebook

Launch Jupyter Notebook and open main.ipynb:

    jupyter notebook main.ipynb

## Usage

In the main.ipynb notebook, you will find the following sections:

    Data Preparation: Loading and preprocessing the transaction data.
    Model Building:
        Decision Tree Classifier: Training and evaluating the Decision Tree model.
        Random Forest Classifier: Training and evaluating the Random Forest model.
    Model Saving: Saving the trained models for future use.
    Model Testing:
        Testing with New Data: Applying the models to new test data for fraud detection.
    Results: Evaluating model performance and displaying results.

## Algorithms and Models

### Decision Tree Classifier:
The Decision Tree model builds a tree-like model of decisions based on data features to classify transactions as suspicious or legitimate.

    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)

### Random Forest Classifier:
The Random Forest model combines multiple decision trees to improve classification accuracy and robustness against overfitting.

    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

## Results:

The notebook includes sections to display:

    Model Accuracy and classification report for decission tree model:
    Accuracy: 1.00
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        17
           1       1.00      1.00      1.00         3

    accuracy                           1.00        20
    macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20

    Model Accuracy and classification report for randomforest model:
    Accuracy: 1.00
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        17
           1       1.00      1.00      1.00         3

    accuracy                           1.00        20
    macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20



## Testing with New Data

To test the models with new data:

    Add Your Test Data to data/test_data.xlsx.
    Run the Testing Cells in the notebook to apply the saved models and view predictions.

Output Generated after testing:
  
    Suspicious detection using Decision tree Algorithm:
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

    Suspicious detection using Random Forest Algorithm:
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

## Future Improvements

    Feature Engineering: Explore additional features to improve model performance.
    Hyperparameter Tuning: Optimize model parameters for better accuracy.
    Model Comparison: Evaluate other algorithms like XGBoost or Support Vector Machines.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
