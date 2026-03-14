# Human Resource ATtrition Rate Analysis using IBM Data
Overview

#This project analyzes employee attrition using machine learning and statistical modeling. It delivers:

End-to-end data cleaning & preprocessing

Exploratory Data Analysis (EDA) with visual insights

Machine learning classification models

Feature importance & ROC/AUC evaluation

A custom HR policy optimization model to minimize predicted attrition

Sensitivity analysis for managerial decision-making

#The goal is to help HR teams identify key drivers of attrition and recommend data-driven HR policy interventions.

#Project Workflow
Load Data → Clean & Transform → EDA → Feature Engineering → 
Train ML Models → Evaluate Performance → 
Optimize HR Policy Decisions → Sensitivity Analysis → Reporting

📂 Repository Structure
📁 HR-Attrition-Analysis
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── scripts/
│   └── hr_attrition_analysis.R
│
├── outputs/
│   ├── plots/
│   ├── model_results/
│   └── optimization/
│
└── README.md
# Tech Stack
Category	Tools
Language	R
Visualization	ggplot2, corrplot
ML Models	Logistic Regression, Decision Tree, XGBoost
Optimization	lpSolve, nloptr
Data Handling	dplyr, tidyr, reshape2
Model Evaluation	caret, pROC, ROCR
#1. Data Preprocessing
 Load dataset & remove irrelevant fields
df <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df <- df %>%
  select(-c(EmployeeCount, StandardHours, Over18, EmployeeNumber))

 Encode target variable

Attrition → binary numeric (classification problem)

df$Attrition <- ifelse(df$Attrition == "Yes", 1, 0)

 Convert categorical variables
df <- df %>% mutate(across(
  c(BusinessTravel, Department, EducationField, Gender,
    JobRole, MaritalStatus, OverTime),
  as.factor
))

2. Exploratory Data Analysis (EDA)
 Missing values analysis
 Summary statistics for all numeric variables
 Outlier detection using Z-Score & IQR
 Visualizations:

Histogram (Age)

Attrition by Gender

Correlation Heatmap
ggplot(df, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue") +
  labs(title = "Histogram of Age")

#3. Machine Learning Models

The attrition prediction problem is binary classification.

Train/Test Split
trainIndex <- createDataPartition(df$Attrition, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test  <- df[-trainIndex, ]
 Remove multicollinearity
num_data <- train %>% select(where(is.numeric))
high_cor <- findCorrelation(cor(num_data), cutoff = 0.75)
train <- train %>% select(-all_of(high_cor))
test  <- test  %>% select(-all_of(high_cor))

Model 1: Logistic Regression
log_model <- glm(Attrition ~ ., data = train, family = "binomial")
log_pred  <- predict(log_model, test, type = "response")
log_class <- ifelse(log_pred > 0.5, 1, 0)
confusionMatrix(as.factor(log_class), as.factor(test$Attrition))


Outputs:

Coefficient table

Confusion Matrix

ROC curve & AUC score

Model 2: Decision Tree
tree_model <- rpart(Attrition ~ ., data = train, method = "class")
tree_pred  <- predict(tree_model, test, type = "class")
confusionMatrix(tree_pred, as.factor(test$Attrition))


Includes pruning & visualization with rpart.plot.

#Model 3: XGBoost
xgb_model <- xgboost(
  data = dtrain,
  objective = "binary:logistic",
  nrounds = 100,
  eta = 0.1,
  max.depth = 3
)


Outputs:

Confusion matrix

AUC score

Feature importance chart

#4. Optimization: Reducing Predicted Attrition

This section uses logistic regression coefficients to optimize HR policies such as:

Salary adjustment

Overtime reduction

Training investment

Youth retention programs

Flexible work options

Optimization setup:
result <- nloptr(
  x0 = x0,
  eval_f = objective_function,
  lb = lb,
  ub = ub,
  eval_g_ineq = eval_g_ineq,
  opts = list("algorithm"="NLOPT_LN_COBYLA", "maxeval"=1000)
)

Output:

Optimal policy decisions

Minimum predicted attrition score

optimal Salary Adjustment: 8.4%
Overtime Reduction: 38.7%
Training Investment: $210
Flexible Work Program: Yes
Minimum Predicted Attrition Score: 0.124

#5. Sensitivity Analysis

A heatmap demonstrates how salary adjustments & overtime reductions jointly affect predicted attrition.

ggplot(grid, aes(x = SalaryAdj*100, y = OvertimeRed*100, fill = AttritionScore)) +
  geom_tile() +
  scale_fill_gradient(low = "green", high = "red")

#Results Summary
XGBoost generally performs best (highest AUC)
Major attrition drivers include:

Overtime

Job Level

Monthly Income

Marital Status

Age

Optimization suggests:

Reduce overtime

Increase flexibility

Targeted training investment

Optional salary adjustments

#Conclusion

This project demonstrates a full machine learning workflow applied to HR analytics.
It not only predicts attrition but also recommends optimized HR strategies supported by quantitative analysis.

This pipeline can be easily extended for:

Cross-department analysis

Scenario simulation

HR dashboards
