# Shooter Talent xG Model

The code here is for creating a model that predicts the probability of an unblocked shot being a goal in a National
Hockey League (NHL) game. The data used for the model spans from the 2007 season until the 2016 season for all regular
season and playoff games. All the data used in this project was scraped using my scraper that can be found
[here.](https://github.com/HarryShomer/Hockey-Scraper) (and is available as a python package as "hockey_scraper").


This is the same as my original xG model (https://github.com/HarryShomer/xG-Model) but this model includes an additional
feature for a player's "shooter talent". This is calculated by using the xG values derived from my original model to
sum previous data for a player and calculate his "shooter multiplier" (Goals/xGoals). This tells us historically how much
better/worse this player is at shooting (as it tells us if he scores more or less than average). This is then regressed
to account for random variation.

# Contact

If you want any more details on how it works you can email me at at Harryshomer@gmail.com.