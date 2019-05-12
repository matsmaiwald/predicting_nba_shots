# Objective
The goal is to predicct whether a given shot from the 2014-15 NBA season is succeful i.e. goes into the basket and scores a point or not.

# Train-test Methodology
I perform an **80-20** train-test sample split and fit different models on the train set. I then evaluate each models performance by using the trained model to predict the outcome of the shots in the test set.

# EDA

## Feature generation

### Original features
For each recored shot, the data contains information on

+ whether the shot was succesful
+ the **player shooting**
+ the **distance to the basket** (= shot distance)
+ whether the shot was worth **two or three points**
+ who the **closest opposing defender** was
+ the **distance to the closest defender**
+ the **matchup/game** (e.g. Boston vs Philadelphia on 04/11/2015)
+ whether the game was a **home or away game** (from the perspective of the player shooter)
+ what the **final margin** of the game was (e.g. +5 if the shooter's team won by five points)
+ the quarter **period of the game**
+ the time on the **game clock**
+ the **number of dribbles** performed by the player immediately before taking the shot


### Additional features

Since close-distance shots tend to have much higher success rates than long-distance shots, I generate two additional features

1. I bin the shot distance into
  + close (0ft to 7ft)
  + semi-close (7ft to 14ft)
  + semi-far (14ft to 23.75ft)
  + far (beyond 23.75ft)
2. For each of the shot distance bins, a Bayesian estimate of the (shooting) player's true shot percentage for that distance bin based on past performance.

### Leveraging Empirical Bayes

Given that shot percentage are constrained to lie between 0 and 1, for each shot distance bin, I fit a $\beta$ distribution on all the shots in the training set in that shot distance bin [INSERT GRAPH HERE]. Using this distribution as a prior and the player's past shot performance in that shot distance bin as the likelihood, I estimate a player's true shot percentage for a given shot distance bin as the mean from the resulting posterior distribution.

Using Bayesian updating to estimate the true shot percentage improves over simpler approaches such as e.g taking the average past success rate, by reducing the noisiness of the resulting feature. Consider for example a player who has just taken three shots from a given shot distance pocket, all of which happened to be unsuccesful. Given that we only have three observations, the Bayesian estimate will lie close to the overall average for that shot pocket, as the prior will mostly overrule the evidence, whereas a simple average would provide an unrealistic estimate of 0%.

## Importance of shot distance and defender distance

### Frequency of shots

### Density difference of shots

# ML models used

## short explanation of each model

# Results

Overview etc

# Next steps

Look at calibration plot
