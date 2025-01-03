# House Price Project

This project aims to predict residential home prices using machine learning techniques. It uses a dataset containing various features of homes, such as size, location, and other attributes, to predict the sale price of a house.

## Project Overview

Predictions of home prices accessible to all can help millions of sellers and buyers engaged in residential real estate transactions. These predictions can help buyers and sellers obtain a deal that reflects the property's value. Consequently, many companies such as Redfin and Zillow work to provide an estimate of residential home prices. To predict the sale price of a house, various features that affect the price are needed, such as the size of the house and location. For example, a larger lot size might lead to an increase in the price of the house, whereas a house next to a busy road might lead to a decrease in the price.

Machine learning can generate predictions for house prices using such features. To this end, machine learning can take input features, find a correlation, and see how they affect the final home price. The correlation between the features can then help generate a final prediction. Therefore, a dataset with training and testing data is required. The training data will train the model to find the correlation between the features, and the testing data will serve as unseen data to evaluate the model.

Accordingly, a dataset that includes such features is needed; however, it is hard to find such a dataset. Conveniently, Kaggle is an online platform that hosts a variety of public datasets. Kaggle has a House Price dataset with features describing different aspects of residential homes in Ames, Iowa. With the dataset, it is possible to construct a model that predicts house prices.
[Kaggle Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Algorithms Used

1. Linear Regression

2. Regularization

3. Gradient Boosting & eXtreme Gradient Boosting (XGBoost)

4. Hyperparameter Tuning

5. Ensemble Learning

## Requirements

- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `scikit-learn`
  - `matplotlib`
  - `plotly`
  - `statsmodels`
