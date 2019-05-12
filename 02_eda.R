remove(list = ls())
# clear console
cat("\014")


library(tidyverse)
library(skimr)
library(ggplot2)

library(corrplot)

# for density plots
library(viridis)
library(rpart.plot)
library(mlbench)
library(rgl)
library(MASS)
library(plot3D)
library(glue)

path_project <- 
  "C:/Users/Mats Ole/Desktop/predicting_nba_shots/"
setwd(path_project)

df_train <- read_csv("./data_output/df_train.csv")

print(glue('{formatC(pct, digits = 2, format = "f")}% of the shots are succesful.',
           pct = 100*mean(df_train$shot_result == "made")))

options(scipen=999)
skim_with(numeric = list(hist = NULL))
skim(df_train)


# Set display ranges to zoom in on areas where vast majority of data points lie
display_range_x <- c(0, 30)
display_range_y <- c(0, 20)

ggplot(df_train,
       aes(x = shot_dist,
           y = closest_defender_dist)
) +
  stat_density2d(aes(fill = ..density..),
                 contour = F, geom = 'tile'
  ) +
  scale_fill_viridis() +
  ggtitle("Close-distance, tightly defended and long-distance, \n less tightly defended shots are the most common") +
  coord_cartesian(xlim = display_range_x,
                  ylim = display_range_y
  ) +
  geom_vline(xintercept = 22,
             linetype = "dotted",
             colour = "red") +
  geom_vline(xintercept = 23.75,
             linetype = "dotted",
             colour = "red") +
  geom_vline(xintercept = 23.75,
             linetype = "dotted",
             colour = "red") +
  annotate("text",
           x = 21,
           y = 15,
           label = paste("Corner 3 points"),
           size = 4,
           angle = 90,
           colour = "red") +
  annotate("text",
           x = 24.75,
           y = 15,
           label = paste("Normal 3 points"),
           size = 4,
           angle = 90,
           colour = "red") +
  xlab("Shot distance in ft") + 
  ylab("Closest defender distance in ft")

ggsave("./Figs/02_eda_shot_density.png", width = 8.3, height = 5.8)

#Short-range shots tend to be more succesful than long-range shots
-----------------------------------------------------------------
  
  #Below, I plot the difference in density between succesful and unsuccesful shots in the shot-distance-closest-defender-space. The graph suggests that shot distance and defender distance will play important rolls in classifying a shot succesful or unsuccesful.


df_made <- filter(df_train, shot_result == "made")
df_missed <- filter(df_train, shot_result == "missed")

n_grid <- 200

common_limits <-
  c(range(df_train$shot_dist),
    range(df_train$closest_defender_dist)
  )

kde_p <- kde2d(df_made$shot_dist,
               df_made$closest_defender_dist,
               n = n_grid,
               lims = common_limits)

kde_n <- kde2d(df_missed$shot_dist,
               df_missed$closest_defender_dist,
               n = n_grid,
               lims = common_limits)

z <- kde_p$z - kde_n$z

png("./Figs/02_eda_shot_density_diff.png")

image2D(x = kde_p$x,
        y = kde_p$y,
        z,
        col = RColorBrewer::brewer.pal(11,"Spectral"),
        xlab = "Shot distance in ft",
        ylab = "Closest defender distance in ft",
        clab = "Density difference",
        shade = 0,
        rasterImage = TRUE,
        xlim = display_range_x,
        ylim = display_range_y,
        contour = list(col = "white",
                       labcex = 0.8,
                       lwd = 1,
                       alpha = 0.5)
)
dev.off()