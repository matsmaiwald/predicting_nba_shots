# Code which explores the NBA shot log data
# clear workspace
remove(list = ls())
# clear console
cat("\014") 

# load packages
library(tidyverse)
library(skimr)
#library(ggplot2)
library(caret)
library(tictoc)
library(corrplot)

path_project <- "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
path_rel_data <- "data_input/"
name_data <- "shot_logs.csv" 

df_raw <- read_csv(paste0(path_project, path_rel_data, name_data))

# data cleaning ----------------------------------------------------------------
df_clean <- df_raw %>% 
  transmute(game_id = GAME_ID,
            matchup = as.factor(MATCHUP),
            home_game = case_when(LOCATION == "H" ~ TRUE,
                                  LOCATION == "A" ~ FALSE),
            win = case_when(W == "W" ~ TRUE,
                           W == "L" ~ FALSE),
            final_margin = abs(FINAL_MARGIN),
            shot_number = SHOT_NUMBER,
            period = PERIOD,
            game_clock = as.numeric(GAME_CLOCK),
            shot_clock = SHOT_CLOCK,
            dribbles = DRIBBLES,
            touch_time = TOUCH_TIME,
            shot_dist = SHOT_DIST,
            pts_type = PTS_TYPE,
            fgm = as.factor(FGM),
            closest_defender = as.factor(CLOSEST_DEFENDER),
            closest_defender_id = CLOSEST_DEFENDER_PLAYER_ID,
            closest_defender_dist = CLOSE_DEF_DIST,
            pts = PTS,
            player_name = as.factor(player_name),
            player_id = player_id) %>%
  drop_na() %>% 
  select(fgm, everything())

# Adding square terms, interactions and dummies
df_clean_2 <- cbind(df_clean[,"fgm"], as.data.frame(model.matrix(fgm ~ - 1 +
                                               shot_clock + 
                                               poly(shot_dist, 2) * pts_type + 
                                               closest_defender_dist + 
                                               final_margin^2 +
                                               touch_time +
                                               #player_name +
                                                 #closest_defender,
                                                 home_game, 
                                             data = df_clean)))

# Data exploration
skim(df_clean_2)
correlations <- cor(df_clean_2 %>% select(-fgm))
corrplot(correlations, order = "hclust")

# check for variables with very low variance
nearZeroVar(df_clean_2)

# pre-process the data ---------------------------------------------------------
set.seed(111)

# create dummy variables and interactions

# stratified random split of the data
df <- df_clean_2[1:100000, ] # only look at part of the data for exploratory analysis
in_training <- createDataPartition(y = df$fgm, p = 0.8, list = FALSE)
df_train <- df[in_training, ]
df_test <- df[-in_training,]

# fit the models ---------------------------------------------------------------

# set up
fitted_models <- list()
model_predictions <- list()
fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
pre_proc_options <- c("center", "scale")
model_spec = fgm ~ .
grids <- list(ridge = expand.grid(alpha = 0,lambda = seq(0, 1, by = 0.2)),
              lasso = expand.grid(alpha = 1,lambda = seq(0, 0.1, by = 0.02)),
              knn = expand.grid(k = seq(1, 81, by = 20))
              )


# ridge
tic()
fitted_models[["ridge"]] <- train(model_spec, data = df_train, 
                                  method = "glmnet", 
                                  preProc = pre_proc_options, 
                                  trControl = fit_control,
                                  tuneGrid = grids[["ridge"]])
model_predictions[["ridge"]] <- predict.train(fitted_models[["ridge"]], df_test)
toc()

# lasso
tic()
fitted_models[["lasso"]] <- train(model_spec, data = df_train, 
                                  method = "glmnet", 
                                  preProc = pre_proc_options, 
                                  trControl = fit_control,
                                  tuneGrid = grids[["lasso"]])
model_predictions[["lasso"]] <- predict.train(fitted_models[["lasso"]], df_test)
toc()

# knn
tic()
fitted_models[["knn"]] <- train(fgm ~ `poly(shot_dist, 2)1` + closest_defender_dist, 
                                data = df_train,
                                method = "knn",
                                preProc = pre_proc_options,
                                trControl = fit_control,
                                tuneGrid = grids[["knn"]])
model_predictions[["knn"]] <- predict.train(fitted_models[["knn"]], df_test)
toc()


# evaluate the benchmark models ------------------------------------------------

confusionMatrix(model_predictions[["ridge"]],df_test$fgm)
varImp(fitted_models[["ridge"]])
plot(fitted_models[["ridge"]])
fitted_models[["ridge"]]
predictors(fitted_models[["ridge"]])
coef(fitted_models[["ridge"]]$finalModel, fitted_models$ridge$bestTune$lambda)

confusionMatrix(model_predictions[["lasso"]],df_test$fgm)
varImp(fitted_models[["lasso"]])
plot(fitted_models[["lasso"]])
fitted_models[["lasso"]]
predictors(fitted_models[["lasso"]])
coef(fitted_models$lasso$finalModel, fitted_models$lasso$bestTune$lambda)

confusionMatrix(model_predictions[["knn"]],df_test$fgm)
#varImp(fitted_models$knn) not implicable to knn
plot(fitted_models[["knn"]])
fitted_models[["knn"]]
predictors(fitted_models[["knn"]])

