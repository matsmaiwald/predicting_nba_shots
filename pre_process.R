# File to preprocess the nba_shot data

library(tidyverse)
library(stats4)
library(ggplot2)
library(lubridate)

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)
path_rel_data_input <- "data_input/"
name_data_input <- "shot_logs.csv"
file_path_data_input <- 
  paste0("./",
    path_rel_data_input,
    name_data_input
    )

df_raw <- read_csv(file_path_data_input)

df_clean <- df_raw %>%
  transmute(game_id = as.factor(GAME_ID),
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
            pts_type = as.factor(PTS_TYPE),
            shot_result = as.factor(SHOT_RESULT),
            closest_defender_name = as.factor(CLOSEST_DEFENDER),
            closest_defender_id = CLOSEST_DEFENDER_PLAYER_ID,
            closest_defender_dist = CLOSE_DEF_DIST,
            pts = PTS,
            player_name = as.factor(player_name),
            player_id = as.factor(player_id),
            date = mdy(str_match(MATCHUP, "^([^-]*?)(-)")[,2])
  ) %>%
  drop_na() %>% 
  select(shot_result, everything()) %>% 
  arrange(date) %>% 
  mutate(observation = row_number(),
         observation_pct = row_number() / max(row_number()))

rm(df_raw)

shot_dist_list <-  
  list(
    close = c(0, 7),
    semi_close = c(7, 14),
    semi_far = c(14, 23.75),
    far = c(23.75, 99)
  )

# Categorise shot distance into 3 bins
df_clean[["shot_dist_cat"]] <- NA_character_
for (distance in names(shot_dist_list)) {
  print(distance)
  df_clean[["shot_dist_cat"]] <- 
    if_else(
      df_clean$shot_dist >= shot_dist_list[[distance]][1] & 
        df_clean$shot_dist <= shot_dist_list[[distance]][2], 
      distance, 
      df_clean[["shot_dist_cat"]] 
    )
}

df_train <- df_clean[df_clean$observation_pct <= 0.8,]
df_test <- df_clean[df_clean$observation_pct > 0.8,]

# Inspect shot percentage per player and distance bucket
df_averages <- 
  df_train %>% 
  group_by(player_id, player_name, shot_dist_cat) %>% 
  summarise(
    succesful_shots = sum(shot_result == "made"),
    attempts = n()
    ) %>%
  mutate(shot_pct = succesful_shots / attempts) %>% 
  ungroup() %>% 
  arrange(shot_dist_cat, desc(shot_pct))


log_likelihood <- function(alpha, beta, attempts, successes) {
  # function to calculate the log-likelihood of a given beta-distribution
  -sum(VGAM::dbetabinom.ab(successes, attempts, alpha, beta, log = TRUE))
}

estimate_alpha_beta <- function(df, successes_col, attempts_col) {
  # returns the mle in a list
  m <- mle(log_likelihood, 
           start = list(alpha = 1, beta = 10), 
           method = "L-BFGS-B",
           fixed = list(successes = df[[successes_col]], attempts = df[[attempts_col]]),
           lower = c(0.0001, .1))
  
  ab <- coef(m)
  params_list <- list(alpha0 = ab[1], beta0 = ab[2])
  params_list
}

# For each distance bin, fit a beta distribution and plot it against the histogram 
# the beta distribution will be the prior for shots from that distance bin
beta_dist_params <- list()
dfs <- list()
for (dist_name in names(shot_dist_list)) {
  dfs[[dist_name]] <- df_averages[df_averages$shot_dist_cat == dist_name,]
  beta_dist_params[[dist_name]] <- estimate_alpha_beta(dfs[[dist_name]], "succesful_shots", "attempts")
  
  p0 = qplot(dfs[[dist_name]]$shot_pct, geom = 'blank') +
    stat_function(fun = dbeta, aes(colour = 'Fitted beta distribution'), 
                  args = list(shape1 = beta_dist_params[[dist_name]]$alpha0, 
                              shape2 = beta_dist_params[[dist_name]]$beta0)
                  ) +
    geom_histogram(aes(y = ..density..), alpha = 0.4) +
    #scale_colour_manual(name = 'Density', values = c('red', 'blue')) +
    theme(legend.position = c(0.85, 0.85)) +
    ggtitle(paste0("Histogram plus fitted beta distribution for ", 
                   dist_name, 
                   " distances"))
  p0
  fig_path <- paste0("./Figs/shot_distribution", dist_name, ".png") 
  ggsave(fig_path)
} 

# calucate estimate of true shot pct as mean of posterior
for (dist_name in names(shot_dist_list)) {
  dfs[[dist_name]] <-
    dfs[[dist_name]] %>% 
    mutate(
      shot_pct_bayesian = 
        (succesful_shots + beta_dist_params[[dist_name]]$alpha0) /
        (attempts + beta_dist_params[[dist_name]]$alpha0 + 
           beta_dist_params[[dist_name]]$beta0)
    )
}

df_combined <- bind_rows(dfs)

df_train <- df_train %>% left_join(df_combined, by = c("player_id", "player_name", "shot_dist_cat"))
df_test <- df_test %>% left_join(df_combined, by = c("player_id", "player_name", "shot_dist_cat"))

write_csv(df_train, "./data_output/df_train.csv")
write_csv(df_test, "./data_output/df_test.csv")
