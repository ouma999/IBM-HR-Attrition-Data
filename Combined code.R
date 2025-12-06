# ==========================================================
# HR Attrition Analysis + Optimization Pipeline
# ==========================================================

# 1. Load Libraries
# ==========================================================
library(tidyverse)
library(dplyr)
library(tidyr)
library(ggplot2)
library(flextable)
library(caret)
library(rpart)
library(rpart.plot)
library(xgboost)
library(pROC)
library(ROCR)
library(corrplot)
library(forcats)
library(reshape2)
library(scales)
library(lpSolve)
library(nloptr)
library(broom)

# ==========================================================
# 2. Load and Clean Data
# ==========================================================
df <- read.csv("C:/Users/polex/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop irrelevant columns
df <- df %>% select(-c(EmployeeCount, StandardHours, Over18, EmployeeNumber))

# Convert target to binary numeric
df$Attrition <- ifelse(df$Attrition == "Yes", 1, 0)

# Convert categorical columns to factors
df <- df %>%
  mutate(
    BusinessTravel = as.factor(BusinessTravel),
    Department = as.factor(Department),
    EducationField = as.factor(EducationField),
    Gender = as.factor(Gender),
    JobRole = as.factor(JobRole),
    MaritalStatus = as.factor(MaritalStatus),
    OverTime = as.factor(OverTime)
  )

# ==========================================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================================
# Missing data summary
missing_data <- colSums(is.na(df))
missing_table <- data.frame(Variable = names(missing_data), Missing = missing_data)
flextable(slice(missing_table, 1:10)) %>% autofit()

# Descriptive statistics for numeric variables
desc_stats <- df %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), list(
    Mean = mean, Median = median, SD = sd, Min = min, Max = max
  ))) %>%
  pivot_longer(everything(), names_sep = "_", names_to = c("Variable", "Statistic")) %>%
  pivot_wider(names_from = Statistic, values_from = value)
flextable(desc_stats) %>% autofit()

# Outlier detection
z_scores <- scale(df %>% select(where(is.numeric)))
outlier_z <- apply(z_scores, 2, function(x) sum(abs(x) > 3))
iqr_outliers <- sapply(df %>% select(where(is.numeric)), function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))
})
outlier_table <- data.frame(
  Variable = names(outlier_z),
  Outliers_Zscore = outlier_z,
  Outliers_IQR = iqr_outliers
)
flextable(tail(outlier_table, 15)) %>% autofit()

# Visualization: Histogram of Age
ggplot(df, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "black") +
  labs(title = "Histogram of Age", x = "Age", y = "Count")

# Attrition by Gender
df_summary <- df %>%
  count(Attrition, Gender) %>%
  mutate(Percentage = n / sum(n) * 100)
ggplot(df_summary, aes(x = Attrition, y = n, fill = Gender)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")), 
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Attrition Rate by Gender", x = "Attrition", y = "Count") +
  theme_minimal()

# Correlation Heatmap
numeric_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(numeric_df, use = "pairwise.complete.obs")
corrplot(corr_matrix, method = "color", tl.cex = 0.8)

# ==========================================================
# 4. Machine Learning Models
# ==========================================================
set.seed(123)
trainIndex <- createDataPartition(df$Attrition, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Multicollinearity removal
num_data <- train %>% select(where(is.numeric))
cor_matrix <- cor(num_data)
high_cor <- findCorrelation(cor_matrix, cutoff = 0.75, names = TRUE)
train <- train %>% select(-all_of(high_cor))
test <- test %>% select(-all_of(high_cor))

# 4a. Logistic Regression
log_model <- glm(Attrition ~ ., data = train, family = "binomial")
log_pred <- predict(log_model, newdata = test, type = "response")
log_class <- ifelse(log_pred > 0.5, 1, 0)
confusionMatrix(as.factor(log_class), as.factor(test$Attrition))
roc_obj <- roc(test$Attrition, log_pred)
plot(roc_obj, col = "blue", main = "ROC Curve - Logistic Regression")
print(paste("AUC:", auc(roc_obj)))
summary(log_model)

# 4b. Decision Tree
tree_model <- rpart(Attrition ~ ., data = train, method = "class")
rpart.plot(tree_model, type = 2, extra = 1, cex = 0.6)
tree_pred <- predict(tree_model, newdata = test, type = "class")
confusionMatrix(tree_pred, as.factor(test$Attrition))
pruned_tree <- prune(tree_model, cp = 0.01)
rpart.plot(pruned_tree, type = 2, extra = 1, cex = 0.7)

# 4c. XGBoost
train_matrix <- model.matrix(Attrition ~ . -1, data = train)
test_matrix <- model.matrix(Attrition ~ . -1, data = test)
dtrain <- xgb.DMatrix(data = train_matrix, label = train$Attrition)
dtest <- xgb.DMatrix(data = test_matrix, label = test$Attrition)
xgb_model <- xgboost(data = dtrain,
                     objective = "binary:logistic",
                     nrounds = 100,
                     max.depth = 3,
                     eta = 0.1,
                     verbose = 0)
xgb_pred <- predict(xgb_model, dtest)
xgb_class <- ifelse(xgb_pred > 0.5, 1, 0)
confusionMatrix(as.factor(xgb_class), as.factor(test$Attrition))
xgb_roc <- roc(test$Attrition, xgb_pred)
plot(xgb_roc, col = "darkgreen", main = "ROC Curve - XGBoost")
print(paste("AUC:", auc(xgb_roc)))
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix = importance, top_n = 10)

# ==========================================================
# 5. Optimization: Minimize Predicted Attrition
# ==========================================================
# Use selected predictors from logistic regression
coef_df <- tidy(log_model) %>%
  mutate(
    estimate = round(estimate, 3),
    std.error = round(std.error, 3),
    statistic = round(statistic, 3),
    p.value = round(p.value, 4)
  )
flextable(coef_df) %>% autofit()

# Objective function: weighted contribution of HR policies
objective_function <- function(x) {
  # x = (salary_adj, overtime_red, training_invest, youth_prog, flexible_work)
  beta_salary <- abs(coef_df$estimate[coef_df$term == "MonthlyIncome"])
  beta_overtime <- abs(coef_df$estimate[coef_df$term == "OverTimeYes"])
  beta_age <- abs(coef_df$estimate[coef_df$term == "Age"])
  beta_marital <- abs(coef_df$estimate[coef_df$term == "MaritalStatusSingle"])
  beta_job <- abs(coef_df$estimate[coef_df$term == "JobLevel"])
  
  attrition_prob <- (beta_salary * (1 - x[1])) + 
    (beta_overtime * (1 - x[2])) + 
    (0.0005 * x[3]) - 
    (0.02 * x[4]) - 
    (0.015 * x[5])
  return(attrition_prob)
}

# Constraints
eval_g_ineq <- function(x) {
  budget_constraint <- (50000 * x[1] + 20000 * (x[3] / 500)) - 200000
  overtime_constraint <- x[2] - 0.5
  equality_constraint <- abs(x[4] + x[5] - 1)
  return(c(budget_constraint, overtime_constraint, equality_constraint))
}

# Optimization run
x0 <- c(0.1, 0.2, 200, 1, 0)
lb <- c(0, 0, 0, 0, 0)
ub <- c(0.2, 0.5, 500, 1, 1)

result <- nloptr(
  x0 = x0,
  eval_f = objective_function,
  lb = lb,
  ub = ub,
  eval_g_ineq = eval_g_ineq,
  opts = list("algorithm" = "NLOPT_LN_COBYLA", "xtol_rel" = 1.0e-8, "maxeval" = 1000)
)

# Print optimization results
cat("Optimal Policy Decisions:\n")
cat("--------------------------------------------------\n")
cat("1. Salary Adjustment (%):", round(result$solution[1]*100, 1), "%\n")
cat("2. Overtime Reduction (%):", round(result$solution[2]*100, 1), "%\n")
cat("3. Training Investment (USD):", round(result$solution[3], 2), "\n")
cat("4. Youth-Focused Retention Program:", ifelse(result$solution[4] > 0.5, "Yes", "No"), "\n")
cat("5. Flexible Work Option:", ifelse(result$solution[5] > 0.5, "Yes", "No"), "\n")
cat("--------------------------------------------------\n")
cat("Minimum Predicted Attrition Score:", round(result$objective, 4), "\n")

# ==========================================================
# 6. Sensitivity Analysis
# ==========================================================
salary_seq <- seq(0, 0.2, by = 0.02)
overtime_seq <- seq(0, 0.5, by = 0.05)
grid <- expand.grid(salary_seq, overtime_seq)
colnames(grid) <- c("SalaryAdj", "OvertimeRed")

grid$AttritionScore <- mapply(function(x1, x2) {
  objective_function(c(x1, x2, result$solution[3], result$solution[4], result$solution[5]))
}, grid$SalaryAdj, grid$OvertimeRed)

ggplot(grid, aes(x = SalaryAdj*100, y = OvertimeRed*100, fill = AttritionScore)) +
  geom_tile() +
  scale_fill_gradient(low = "darkgreen", high = "red") +
  labs(
    title = "Sensitivity Analysis: Salary vs Overtime Impact on Attrition",
    x = "Salary Adjustment (%)",
    y = "Overtime Reduction (%)",
    fill = "Attrition Score"
  ) +
  theme_minimal()
