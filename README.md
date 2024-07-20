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

## Usage
1. Load the dataset:
    ```python
    import pandas as pd

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    ```

2. Visualize the data:
    ```python
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], alpha=0.5)
    plt.title('House Price vs. Above Ground Living Area')
    plt.xlabel('Above Ground Living Area (sq ft)')
    plt.ylabel('Sale Price ($)')
    plt.show()
    ```

3. Preprocess the data and train the model:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    target = 'SalePrice'

    X = train_data[features]
    y = train_data[target]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, features)
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    ```

4. Make predictions on the test set:
    ```python
    X_test = test_data[features]
    test_preds = model.predict(X_test)

    submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())
    ```

5. Predict house price based on user input:
    ```python
    def predict_house_price(gr_liv_area, bedrooms, full_bath, tot_rms_abv_grd, year_built):
        input_data = pd.DataFrame({
            'GrLivArea': [gr_liv_area],
            'BedroomAbvGr': [bedrooms],
            'FullBath': [full_bath],
            'TotRmsAbvGrd': [tot_rms_abv_grd],
            'YearBuilt': [year_built]
        })
        predicted_price = model.predict(input_data)
        return predicted_price[0]

    user_gr_liv_area = 2000
    user_bedrooms = 3
    user_full_bath = 2
    user_tot_rms_abv_grd = 7
    user_year_built = 1995

    predicted_price = predict_house_price(user_gr_liv_area, user_bedrooms, user_full_bath, user_tot_rms_abv_grd, user_year_built)
    print(f'Predicted House Price: ${predicted_price:.2f}')
    ```

## Visualization
The project includes visualizations to understand the relationship between the features and the house prices. You can find the visualization code in the `usage` section above.

## Prediction
The project provides a function to predict house prices based on user input for features such as living area, number of bedrooms, number of bathrooms, total rooms above ground, and year built.

## Results
- The model's performance is evaluated using Mean Squared Error (MSE) and R^2 Score.
- Example predictions and submission files are generated for the test set.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
