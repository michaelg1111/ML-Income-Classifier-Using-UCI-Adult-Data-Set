# ML-Income-Classifier-Using-UCI-Adult-Data-Set

This project involves building a machine learning model to predict whether an individual’s income exceeds $50K based on the UCI Adult Data Set. The analysis uses various machine learning algorithms, including logistic regression, random forest, and lasso regression.

## Project Steps:
1. **Data Loading**: The UCI Adult Data Set is loaded into R.
2. **Data Cleaning**: The dataset is cleaned by handling missing values and transforming categorical variables.
3. **Feature Engineering**: New features, such as the square of age, are added.
4. **Modeling**: Various machine learning models are trained to predict income, including:
   - Lasso regression
   - Ridge regression
   - Random forest
5. **Model Evaluation**: Models are evaluated using classification accuracy, confusion matrices, and other metrics.

## R Code:
- The code includes steps for data cleaning, feature engineering, and model evaluation.
- Use R’s `caret` and `randomForest` packages to build and evaluate the models.

## Dependencies:
- `caret` package
- `randomForest` package
- `dplyr` package
