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

fit_control <- trainControl(method = "cv",
                            number = 3,
                            classProbs = TRUE,
                            summaryFunction = mnLogLoss)

pre_proc_options <- c("center", "scale")

optimisation_metric <- "logLoss"

grids <- list()
model_specs <- list()


# logistic regression-----------------------------------------------------------

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
  home_game +
  shot_pct_bayesian +
  shot_dist:shot_pct_bayesian


tic()
grids$lasso <- expand.grid(alpha = 1,lambda = 10 ^ seq(-4, 0, length = 5))
set.seed(111)
fitted_models[["lasso"]] <- train(model_specs$regression,
                                  data = df_train,
                                  method = "glmnet",
                                  metric = optimisation_metric,
                                  preProc = pre_proc_options,
                                  trControl = fit_control,
                                  tuneGrid = grids[["lasso"]]#,
                                  #verboseIter = T
)
model_predictions[["lasso"]] <- predict.train(fitted_models[["lasso"]], df_test)
toc()

plot(fitted_models$lasso)

# RF ---------------------------------------------------------------------------
model_specs$rf <- shot_result ~ shot_dist +
  pts_type +
  closest_defender_dist +
  shot_clock +
  shot_pct_bayesian +
  touch_time +
  final_margin +
  touch_time +
  dribbles +
  home_game

grids$rf <- expand.grid(mtry = c(1, 2, 3), splitrule = "gini", min.node.size = 100)

tic()
set.seed(111)
fitted_models[["rf"]] <- train(model_specs$rf, data = df_train,
                               method = "ranger",
                               metric = optimisation_metric,
                               #tuneLength = 1,
                               trControl = fit_control,
                               num.trees = 500,
                               tuneGrid = grids[["rf"]]
)
toc()
model_predictions[["rf"]] <- predict.train(fitted_models[["rf"]], df_test)
plot(fitted_models$rf)

# xgboost ----------------------------------------------------------------------
model_specs$xgb <- shot_result ~ shot_dist +
  pts_type +
  closest_defender_dist +
  shot_clock +
  shot_pct_bayesian +
  touch_time +
  final_margin +
  touch_time +
  dribbles +
  home_game

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


# SVM --------------------------------------------------------------------------
model_specs$svm <- shot_result ~ shot_dist +
  pts_type +
  closest_defender_dist +
  shot_clock +
  shot_pct_bayesian +
  touch_time +
  final_margin +
  touch_time +
  dribbles +
  home_game

grids$svm <- expand.grid(sigma = 2^c(-10, -5), C = 10 ^ seq(-2, 1, length = 4))

tic()
set.seed(111)
fitted_models[["svm"]] <- train(model_specs$svm, data = df_train[1:1000,],
                                method = "svmRadial",
                                metric = optimisation_metric,
                                preProc = pre_proc_options,
                                trControl = fit_control,
                                tuneGrid = grids[["svm"]]
)
toc()

model_predictions[["svm"]] <- predict.train(fitted_models[["svm"]], df_test)
plot(fitted_models$svm)

results <-
  resamples(
    list(
      LogisticLasso = fitted_models[["lasso"]],
      RF = fitted_models[["rf"]],
      XGB = fitted_models[["xgb"]],
      SVM = fitted_models[["svm"]]
      )
    )

save.image(file = "./train_predict.RData")

# boxplots of results-----------------------------------------------------------
summary(results)

png("./Figs/03_train_predict_model_comparison.png")
bwplot(results)
dev.off()

make_calibration_plot <- function(df, fitted_model) {
  df$probs <- predict.train(fitted_model, df, type = "prob")[,1]
  calibration_curve <- calibration(shot_result ~ probs, data = df)
  xyplot(calibration_curve, auto.key = list(columns = 2))
}

# Evaluation on test set--------------------------------------------------------

png("./Figs/03_train_predict_calibration_xgb.png")
make_calibration_plot(df_test, fitted_models$xgb)
dev.off()

confusionMatrix(model_predictions$xgb,as.factor(df_test$shot_result), positive = "made")
