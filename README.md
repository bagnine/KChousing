# King's County Housing Project

## The Premise

Using a large data set of houses sold in the greater Seattle area between 2014 and 2015, I've been asked to predict the sale price of a house with a similar set of attributes by creating a multilinear regression model.

### Features and Questions

* What are the most important variables in predicting sale price?
* Is there a way to reformat or combine data to improve the model?
* What type of statistical relationships will be most beneficial to the model?

## The Process

### Exploratory Data Analysis

First, I created some visualizations to determine whether variables were continuous or categorical, and their correlations with my target variable (sale price).
![Dataset Histograms](./Images/hists.png)
![Dataset Correlation Heatmap](./Images/heatmap.png)

After running some statistical tests, I had a good idea of which features were important and which ones I would need to transform in order to be useful.

After a little bit of cleaning to remove outliers and erroneous data, I was ready to move on to feature engineering.

### Feature Engineering

Several of the variables in the dataset are categorical and a few (date and zipcode) look like numbers to the linear regression model and need to be modified in order to fit the model. After performing ANOVA tests on several of them, I modified the dates into months (because the span of the data is only 1 year and I didn't find a big difference between May of 2014 and 2015 I just used the month) and created dummy columns for several variables including grade, view and zipcode.

![Regression Plots](./Images/regplots.png)

After creating scatterplots with regression lines for many of the continuous features with price, I noticed several had a slight curve or a regression line that didn't seem to fit the data well. I performed log transformations on a few to redistribute the data and hopefully better fit my model. 

I thought school districts might be a good predictor of home value, so I Googled best school districts in Seattle and made another dummy column for the top ten. After doing several A/B tests of the model, I dropped the list of top school zipcodes to the top four, which had a bigger impact on my model.

### Model Testing

I tested several different models with various combinations of the existing features and a few I created. Most of the aggregated variables made my model worse, and overall I'd say a less is more approach served me best. The dummies of the categoricals improved the model to differing degrees with zipcode being the most impactful. School districts did almost nothing.

After putting together a few features that seemed to work best, I plotted out the residuals to see how they were distributed. The distribution was heteroscedastic, and I decided to try to fix that through Polynomial Functions.

![Heteroscedastic Scatter](./Images/scatter.png)

After a few tests of polynomial variations of my model, I found a fit that was near my goals. The residual plot was a lot more evenly distributed, indicating that I managed to compensate for more of the relationship. 

![Homoscedastic Scatter](./Images/scatter2.png)

Finally, I plotted the price data from my test and training set along with the predictions.

![Prediction distribution plot](./Images/distplot.png)

Pretty good but not perfect. There are still some outliers to account for, but the model is performing well. 

## The Conclusion

I was able to get an RMSE of 107,000. Not quite my goal of 100k, but still close. 

With a bit more time, I'd go back into the data and review the distributions and outliers. I think my model is solid and there are likely things in the data that are lowering its predictive power. 

## Repository structure

```
├── README.md                           <- This README file
├── code
│   ├── housing.ipynb                   <- Exploratory notebook
│   ├── final.ipynb                     <- Notebook with final model and pickle
│   ├── data_clean.py                   <- .py functions for data cleaning
|   ├── model.py                        <- .py functions for testing linear regression models
├── data                                <- The housing dataset and holdout
└── images                              <- Visualizations used in README
