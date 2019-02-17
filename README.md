# Classifying NBA shots

In this notebook, three ML models are used to predict whether a basketball shot is successful (hits the basket and scores a point) or not. Specifically, we'll look at all the basketball shots made during the 2014-2015 NBA season scraped from the NBA's API and provided on kaggle [(https://www.kaggle.com/dansbecker/nba-shot-logs/home)](https://www.kaggle.com/dansbecker/nba-shot-logs/home).


Given the stochastic nature of basketball shots -- a given shot that was succesful may not be succesful again when repeated under the exact same match conditions, classifying NBA shots proves rather difficult. The model has particular difficulty to identify succesful shots.

TO DO

1.  Enlarge feature space
    -   Group data along the distances of the shot and the closest defender
    -   Find players that outperform their peers in a given shot group
2.  Refit Logistic Regression model plus MARS and Random Forrests

The output of the notebook can be found in the [main.md](main.md) file. The full code is located in the [main.Rmd](main.Rmd) file.
