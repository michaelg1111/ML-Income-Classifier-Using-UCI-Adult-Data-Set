##################################################
# ECON 418-518 Homework 3
# Michael Gunderson
# The University of Arizona
# michaelgunderson@arizona.edu 
# 17 November 2024
###################################################
#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set sead
set.seed(418518)

# Set working directory to the folder containing your data set
setwd("/Users/gg/Desktop/adult")

# Load the data set into R as a data table
data <- read.csv("ECON_418-518_HW3_Data.csv")

# Preview the data
head(data)

#####################
# Problem 1
#####################
#################
# Question (i)
#################
# Dropping specified columns from table
data <- data[, !(names(data) %in% c("fnlwgt", "occupation", "relationship", "capital-gain", "capital-loss", "educational-num"))]

#################
# Question (ii)
#################
##############
# Part (a)
##############
# Convert the income column to binary: 1 for ">50K", 0 otherwise
data$income <- ifelse(data$income == ">50K", 1, 0)

##############
# Part (b)
##############
# Convert race to binary: 1 if "White", 0 otherwise
data$race <- ifelse(data$race == "White", 1, 0)

##############
# Part (c)
##############
# Convert gender to binary: 1 if "Male", 0 otherwise
data$gender <- ifelse(data$gender == "Male", 1, 0)

##############
# Part (d)
##############
# Convert workclass to binary: 1 if "Private", 0 otherwise
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

##############
# Part (e)
##############
# Convert native country to binary: 1 if "United-States", 0 otherwise
data$native.country <- ifelse(data$native.country == "United-States", 1, 0)

##############
# Part (f)
##############
# Convert marital status to binary: 1 if "Married-civ-spouse", 0 otherwise
data$marital.status <- ifelse(data$marital.status == "Married-civ-spouse", 1, 0)

##############
# Part (g)
##############
# Convert education to binary: 1 if "Bachelors", "Masters", or "Doctorate", 0 otherwise
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)


##############
# Part (h)
##############
# Create age squared variable
data$age_sq <- data$age^2

##############
# Part (i)
##############
# Standardize age, age squared, and hours per week variables
standardize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

data$age <- standardize(data$age)
data$age_sq <- standardize(data$age_sq)
data$hours.per.week <- standardize(data$hours.per.week)

#################
# Question (iii)
#################

##############
# Part (a)
##############
# Proportion of individuals with income > $50,000
prop_income_over_50k <- mean(data$income == 1)
prop_income_over_50k

##############
# Part (b)
##############
# Proportion of individuals in the private sector
prop_private_sector <- mean(data$workclass == 1)
prop_private_sector

##############
# Part (c)
##############
# Proportion of married individuals
prop_married <- mean(data$marital.status == 1)
prop_married

##############
# Part (d)
##############
# Proportion of females in the data set (gender = 0)
prop_females <- mean(data$gender == 0)
prop_females

##############
# Part (e)
##############
# Total number of NAs in the data set
total_NAs <- sum(is.na(data))
total_NAs

##############
# Part (f)
##############
# Convert income variable to a factor
data$income <- as.factor(data$income)

# Verify the conversion
str(data$income)

#################
# Question (iv)
#################

##############
# Part (a)
##############
# Calculate the index of the last training set observation
last_train_index <- floor(nrow(data) * 0.70)

##############
# Part (b)
##############
# Create the training data table
training_data <- data[1:last_train_index, ]

##############
# Part (c)
##############
# Create the testing data table
testing_data <- data[(last_train_index + 1):nrow(data), ]

cat("Training set size:", nrow(training_data), "\n")
cat("Testing set size:", nrow(testing_data), "\n")
cat("Total size:", nrow(data), "\n")

#################
# Question (v)
#################
##############
# Part (b)
##############
# Estimating a Lasso Regression Model using caret's train() function with 10-fold cross-validation

install.packages("caret")
install.packages("glmnet")
library(caret)
library(glmnet)

# Define the grid of lambda values
lambda_grid <- 10^(seq(2, -2, length = 50))

# Train a lasso regression model with 10-fold cross-validation
set.seed(418518)
lasso_model <- train(
  income ~ ., 
  data = training_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Example: train the model (adjust this based on your actual model setup)
cv <- train(income ~ ., data = training_data, method = "glmnet", 
            tuneGrid = expand.grid(alpha = 1, lambda = seq(10^5, 10^-2, length = 50)), 
            trControl = trainControl(method = "cv", number = 10))

# Display the best lambda and accuracy
lasso_model$bestTune
max(lasso_model$results$Accuracy)

##############
# Part (c)
##############
# Best value of lambda and classification accuracy

best_lambda <- lasso_model$bestTune$lambda
classification_accuracy <- max(lasso_model$results$Accuracy)

cat("Best lambda for Lasso:", best_lambda, "\n")
cat("Classification accuracy for Lasso:", classification_accuracy, "\n")

##############
# Part (d)
##############
# Variables with coefficients approximately zero

lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
zero_coeff_vars <- rownames(lasso_coefficients)[abs(as.vector(lasso_coefficients)) < 1e-4]

cat("Variables with zero coefficients in Lasso:", zero_coeff_vars, "\n")

##############
# Part (e)
##############
# Estimate Lasso and Ridge regression models with only non-zero coefficient variables

# Filter out non-zero variables from training data
non_zero_vars <- setdiff(colnames(training_data), zero_coeff_vars)
filtered_training_data <- training_data[, c(non_zero_vars, "income")]

# Lasso Model with non-zero variables
set.seed(418518)
lasso_refined <- train(
  income ~ ., 
  data = filtered_training_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Ridge Model with non-zero variables
set.seed(418518)
ridge_refined <- train(
  income ~ ., 
  data = filtered_training_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)
)

# Compare Classification Accuracies
lasso_accuracy <- max(lasso_refined$results$Accuracy)
ridge_accuracy <- max(ridge_refined$results$Accuracy)

cat("Lasso Classification Accuracy:", lasso_accuracy, "\n")
cat("Ridge Classification Accuracy:", ridge_accuracy, "\n")

#################
# Question (vi)
#################
install.packages("randomForest")
##############
# Part (b)
##############
cat("Starting Random Forest Model Evaluation...\n")

library(caret)
library(randomForest)

# Define the grid for tuning mtry (number of features to try at each split)
tree_grid <- expand.grid(mtry = c(2, 5, 9))  # Adjust as needed based on features

# Train Random Forest models with 5-fold cross-validation and different number of trees
rf_model_100 <- train(
  income ~ ., 
  data = training_data, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 100  # 100 trees
)

rf_model_200 <- train(
  income ~ ., 
  data = training_data, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 200  # 200 trees
)

rf_model_300 <- train(
  income ~ ., 
  data = training_data, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 300  # 300 trees
)

cat("Random Forest Models Complete!\n")
##############
# Part (c)
##############
cat("Evaluating Model Accuracy...\n")

# Print the results to compare the models' accuracy
cat("Model 100 trees:", rf_model_100$results$Accuracy, "\n")
cat("Model 200 trees:", rf_model_200$results$Accuracy, "\n")
cat("Model 300 trees:", rf_model_300$results$Accuracy, "\n")

# Find the best model based on accuracy
best_rf_model <- max(rf_model_100$results$Accuracy, rf_model_200$results$Accuracy, rf_model_300$results$Accuracy)
cat("Best Model Accuracy:", best_rf_model, "\n")
##############
# Part (d)
##############
cat("Comparing Best Models...\n")

# Assuming you have the best Random Forest model accuracy from Part (vi) saved
best_rf_model <- 0.8570592  
# Define the best accuracy for the Lasso/Ridge model (from Part (v))
# Assuming 'cv' is your lasso/ridge model training object, extract the best accuracy:
best_lasso_ridge_accuracy <- max(cv$results$Accuracy)  

# Print comparison
cat("Best Random Forest Model Accuracy:", best_rf_model, "\n")
cat("Best Lasso/Ridge Model Accuracy from Part (v):", best_lasso_ridge_accuracy, "\n")
##############
# Part (e)
##############
#################
# Confusion Matrix
#################
cat("Generating Confusion Matrix...\n")

# Make predictions using the best random forest model 
predictions_rf <- predict(rf_model_100, training_data)

# Generate the confusion matrix
cm_rf <- confusionMatrix(predictions_rf, training_data$income)

# Print confusion matrix
print(cm_rf)

# Check false positives and false negatives
cat("False Positives:", cm_rf$table[2,1], "\n")
cat("False Negatives:", cm_rf$table[1,2], "\n")

#################
# Question (vii)
#################
cat("Evaluating Best Model on Testing Data...\n")

# Make predictions using the best random forest model (example: rf_model_100)
predictions_rf_test <- predict(rf_model_100, testing_data)

# Evaluate the classification accuracy on the testing set
test_accuracy_rf <- mean(predictions_rf_test == testing_data$income)
cat("Classification Accuracy on Testing Set:", test_accuracy_rf, "\n")

# This classification accuracy represents the percentage of correct predictions made by the model on the testing data.
