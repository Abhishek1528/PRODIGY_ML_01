# PRODIGY_ML_01

# House Price Prediction using Linear Regression

This project demonstrates how to build a linear regression model to predict house prices based on several features such as square footage, number of bedrooms, and number of bathrooms. It also includes visualizations of the data and an interactive user input function for predicting house prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Prediction](#prediction)
- [Results](#results)
- [License](#license)

## Project Overview
This project uses a dataset of house prices to train a linear regression model. The model is used to predict house prices based on features such as the living area, number of bedrooms, number of bathrooms, and other relevant attributes. The project also provides visualizations to understand the relationship between these features and house prices.

## Dataset
The dataset consists of the following files:
- `train.csv`: The training set.
- `test.csv`: The test set.
- `data_description.txt`: Full description of each column.
- `sample_submission.csv`: A benchmark submission.

## Requirements
- Python 3.6+
- pandas
- matplotlib
- scikit-learn

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/house-price-prediction.git
    cd house-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
    
## Visualization
The project includes visualizations to understand the relationship between the features and the house prices. You can find the visualization code in the `usage` section above.

## Prediction
The project provides a function to predict house prices based on user input for features such as living area, number of bedrooms, number of bathrooms, total rooms above ground, and year built.

## Results
- The model's performance is evaluated using Mean Squared Error (MSE) and R^2 Score.
- Example predictions and submission files are generated for the test set.

