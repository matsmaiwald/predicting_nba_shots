# File to preprocess the nba_shot data

library(tidyverse)
library(stats4)
library(ggplot2)

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)
path_rel_data <- "data_input/"
name_data <- "shot_logs.csv"
data_file_path <- 
  paste0("./",
    path_rel_data,
    name_data
    )

df_raw <- read_csv(data_file_path)

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
            player_id = as.factor(player_id)
  ) %>%
  select(shot_result, everything())
rm(df_raw)

df_averages <- 
  df_clean %>% 
  group_by(player_id, player_name) %>% 
  summarise(
    succesful_shots = sum(shot_result == "made"),
    attempts = n()
    ) %>%
  mutate(shot_pct = succesful_shots / attempts) %>% 
  ungroup() %>% 
  filter(attempts > 300)

log_likelihood <- function(alpha, beta, attempts, successes) {
  -sum(VGAM::dbetabinom.ab(successes, attempts, alpha, beta, log = TRUE))
}

m <- mle(log_likelihood, 
         start = list(alpha = 1, beta = 10), 
         method = "L-BFGS-B",
         fixed = list(successes = df_averages$succesful_shots, attempts = df_averages$attempts),
         lower = c(0.0001, .1))

ab <- coef(m)
alpha0 <- ab[1]
beta0 <- ab[2]


p0 = qplot(df_averages$shot_pct, geom = 'blank') +   
  stat_function(fun = dbeta, aes(colour = 'Normal'), args = list(shape1 = alpha0, shape2 = beta0)) +                       
  geom_histogram(aes(y = ..density..), alpha = 0.4) +                        
  #scale_colour_manual(name = 'Density', values = c('red', 'blue')) + 
  theme(legend.position = c(0.85, 0.85))
p0


# -----------------------------------------------------------------------------
