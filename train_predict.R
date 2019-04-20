library(caret)
library(tidyverse)
library(tictoc)
library(pROC)
library(ranger)
library(xgboost)

rm(list = ls())

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)

df_train <- read_csv("./data_output/df_train.csv") %>% drop_na()
df_test <- read_csv("./data_output/df_test.csv") %>% drop_na()

# set up
fitted_models <- list()
model_predictions <- list()
# fit_control <- trainControl(method = "cv",
#                             number = 3,
#                             classProbs = TRUE,
#                             summaryFunction = twoClassSummary)

fit_control <- trainControl(method = "cv",
                            number = 3,
                            classProbs = TRUE,
                            summaryFunction = mnLogLoss)

pre_proc_options <- c("center", "scale")

#optimisation_metric <- "ROC"
optimisation_metric <- "logLoss"

make_roc_plot <- function(df, fitted_model) {
  df$probs <- predict.train(fitted_model, df, type = "prob")[,2]
  roc_curve <- roc(response = df$shot_result, predictor = df$probs)
  print(auc(roc_curve))
  plot(roc_curve, legacy.axes = TRUE)
}

make_calibration_plot <- function(df, fitted_model) {
  df$probs <- predict.train(fitted_model, df, type = "prob")[,1]
  calibration_curve <- calibration(shot_result ~ probs, data = df)
  xyplot(calibration_curve, auto.key = list(columns = 2))
}

grids <- list()
model_specs <- list()

# regression

model_specs$regression <-
  shot_result ~ shot_dist*pts_type +
  I(shot_dist^2) +
  I(shot_dist^2):pts_type +
  closest_defender_dist*pts_type +
  I(closest_defender_dist^2) +
  I(closest_defender_dist^2):pts_type +
  closest_defender_dist:shot_dist +
  shot_clock +
  final_margin +
  touch_time +
  dribbles +
  home_game #+
  #shot_pct_bayesian +
  #shot_dist:shot_pct_bayesian


tic()
grids$lasso <- expand.grid(alpha = 1,lambda = 10 ^ seq(-6, -2, length = 5))
set.seed(111)
fitted_models[["lasso"]] <- train(model_specs$regression, 
                                  data = df_train,
                                  method = "glmnet",
                                  metric = optimisation_metric,
                                  preProc = pre_proc_options,
                                  trControl = fit_control,
                                  tuneGrid = grids[["lasso"]]
)
model_predictions[["lasso"]] <- predict.train(fitted_models[["lasso"]], df_test)
toc()

plot(fitted_models$lasso)
confusionMatrix(model_predictions$lasso,as.factor(df_test$shot_result), positive = "made")
varImp(fitted_models$lasso)

# RF -------------------------------------------
model_specs$rf <- shot_result ~ shot_dist + pts_type + 
  closest_defender_dist + 
  shot_clock +
  shot_pct_bayesian +
  shot_clock +
  touch_time

grids$rf <- expand.grid(mtry = c(1, 2), splitrule = "gini", min.node.size = 100)

tic()
set.seed(111)
fitted_models[["rf"]] <- train(model_specs$rf, data = df_train,
                               method = "ranger",
                               metric = optimisation_metric,
                               #tuneLength = 1,
                               trControl = fit_control,
                               num.trees = 100,
                               tuneGrid = grids[["rf"]]
)
toc()
model_predictions[["rf"]] <- predict.train(fitted_models[["rf"]], df_test)
plot(fitted_models$rf)
confusionMatrix(model_predictions$rf,as.factor(df_test$shot_result), positive = "made")
make_roc_plot(df_test, fitted_models$rf)
make_calibration_plot(df_test, fitted_models$svm)

# xgboost ------------------------------------------------------------------
model_specs$xgb <- shot_result ~ shot_dist + pts_type + 
  closest_defender_dist + 
  shot_clock +
  shot_pct_bayesian +
  shot_clock +
  touch_time

grids$xgb <- expand.grid(nrounds = c(50,100,300, 500), 
                         max_depth = 5, 
                         eta = c(0.01, 0.02),
                         gamma = 0.01,
                         colsample_bytree = c(0.5),
                         min_child_weight = 0,
                         subsample = 0.8)

tic()
set.seed(111)
fitted_models[["xgb"]] <- train(model_specs$xgb, data = df_train,
                               method = "xgbTree",
                               metric = optimisation_metric,
                               #tuneLength = 1,
                               trControl = fit_control,
                               tuneGrid = grids[["xgb"]]
)
toc()

model_predictions[["xgb"]] <- predict.train(fitted_models[["xgb"]], df_test)
plot(fitted_models$xgb)
confusionMatrix(model_predictions$xgb,as.factor(df_test$shot_result), positive = "made")
make_roc_plot(df_test, fitted_models$xgb)


# SVM ----------------------------------------------------------------------
model_specs$svm <- shot_result ~ shot_dist + pts_type + 
  closest_defender_dist + 
  shot_clock +
  shot_pct_bayesian +
  shot_clock +
  touch_time

grids$svm <- expand.grid(sigma = 2^c(-10, -5), C = 10 ^ seq(-2, 1, length = 4))

tic()
set.seed(111)
fitted_models[["svm"]] <- train(model_specs$svm, data = df_train[1:10000,],
                                method = "svmRadial",
                                metric = optimisation_metric,
                                preProc = pre_proc_options,
                                trControl = fit_control,
                                tuneGrid = grids[["svm"]]
)
toc()

model_predictions[["svm"]] <- predict.train(fitted_models[["svm"]], df_test)
plot(fitted_models$svm)
confusionMatrix(model_predictions$svm,as.factor(df_test$shot_result), positive = "made")
make_roc_plot(df_test, fitted_models$svm)


results <- 
  resamples(
    list(
      LogisticLasso = fitted_models[["lasso"]], 
      RF = fitted_models[["rf"]], 
      XGB = fitted_models[["xgb"]],
      SVM = fitted_models[["svm"]]
      )
    )

# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)

save.image(file = "./train_predict.RData")

make_calibration_plot(df_test, fitted_models$lasso)
make_calibration_plot(df_test, fitted_models$rf)
make_calibration_plot(df_test, fitted_models$xgb)
make_calibration_plot(df_test, fitted_models$svm)
test <- cbind(df_test, model_predictions$xgb)
brierscore(shot_result ~ model_predictions$xgb, data = cbind(df_test, model_predictions$xgb))

calculate_brier_score <- function(df, fitted_model) {
  #browser()
  df$predictions <- round(predict.train(fitted_model, df, type = "prob")[,1], 2)
  df$actuals <- if_else(df$shot_result == "made", 1, 0)
  #df$predictions <- 0
  score_list <- brierscore(actuals ~ predictions, data = df, group = "predictions")
  df$difff <- score_list$rawscores
  mean(score_list$rawscores)
}

result <- calculate_brier_score(df_test, fitted_models$lasso)
result <- calculate_brier_score(df_test, fitted_models$xgb)
result <- calculate_brier_score(df_test, fitted_models$rf)
result <- calculate_brier_score(df_test, fitted_models$svm)
