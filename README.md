# Turbofan-RUL-Predictive-Maintenance-Project 

## Project Overview
This project develops a machine learning solution for predicting the health condition of turbofan engines using predictive maintenance techniques. By analyzing sensor data, the model classifies engine conditions into three states: Good, Moderate, and Warning.

## Key Features
- Preprocessing of turbofan sensor data
- Multi-class classification of engine health
- Implementation of multiple machine learning algorithms
- Cross-validation and hyperparameter tuning

## Methodology
### Data Preprocessing
- Load sensor data from text files
- Remove constant-value sensors
- Calculate Life Ratio (LR) to define engine condition
- Label engine states:
  - Good Condition (LR <= 0.6)
  - Moderate Condition (0.6 < LR <= 0.8)
  - Warning Condition (LR > 0.8)

### Machine Learning Models
Implemented classifiers include:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- LightGBM Classifier
- XGBoost Classifier

### Model Evaluation
- Train-test split (80/20)
- Accuracy score as primary metric
- Stratified K-Fold cross-validation
- Randomized hyperparameter search

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- lightgbm
- xgboost

## Installation

1. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Performance
Best model achieved accuracy around ~96% using XGBoost with hyperparameter tuning.

## Future Work
- Implement more sophisticated feature engineering
- Explore deep learning architectures
- Develop real-time prediction capabilities.
