library(scoring)
library(caret)
library(dplyr)
library(pROC)

rm(list = ls())

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)

load("./train_predict.RData")

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

calculate_brier_score <- function(df, fitted_model) {
  #browser()
  df$predictions <- round(predict.train(fitted_model, df, type = "prob")[,1], 2)
  df$actuals <- if_else(df$shot_result == "made", 1, 0)
  #df$predictions <- 0
  score_list <- brierscore(actuals ~ predictions, data = df, group = "predictions")
  df$difff <- score_list$rawscores
  mean(score_list$rawscores)
}

# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
brier_scores <- list()
for (model_name in c("lasso", "xgb", "rf", "svm")) {
  #model_name <- "xgb"
  brier_scores[[model_name]] <- 
    calculate_brier_score(df_test, fitted_models[[model_name]])
  make_calibration_plot(df_test, fitted_models[[model_name]])
  ggsave(paste0("./Figs/calibration_plot_", model_name, ".png"))
  make_roc_plot(df_test, fitted_models[[model_name]])
  ggsave(paste0("./Figs/roc_plot_", model_name, ".png"))
  
}
