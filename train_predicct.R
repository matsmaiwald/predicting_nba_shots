library(caret)
library(tidyverse)
library(tictoc)
library(pROC)

rm(list = ls())

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)

df_train <- read_csv("./data_output/df_train.csv") %>% drop_na()
df_test <- read_csv("./data_output/df_test.csv") %>% drop_na()

# set up
fitted_models <- list()
model_predictions <- list()
fit_control <- trainControl(method = "cv",
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)
pre_proc_options <- c("center", "scale")
optimisation_metric <- "ROC"

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
  #shot_clock +
  final_margin +
  touch_time +
  dribbles +
  home_game +
  shot_pct_bayesian +
  shot_dist:shot_pct_bayesian


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

make_roc_plot <- function(df, fitted_model) {
  df$probs <- predict.train(fitted_model, df, type = "prob")[,2]
  roc_curve <- roc(response = df$shot_result, predictor = df$probs)
  print(auc(roc_curve))
  plot(roc_curve, legacy.axes = TRUE)
}

make_roc_plot(df_test, fitted_models$lasso)
