import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def main():
    df = pd.read_csv('Ecommerce Customers.csv')

    print('Head:')
    print(df.head())
    print('
Info:')
    print(df.info())
    print('
Describe:')
    print(df.describe())
    print('
Missing values:')
    print(df.isnull().sum())
    print('
Duplicated rows:', df.duplicated().sum())

    X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = df['Yearly Amount Spent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

    print('
Intercept:')
    print(model.intercept_)
    print('
Coefficients:')
    print(coeff_df)
    print('
Evaluation:')
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('R2 Score:', metrics.r2_score(y_test, predictions))


if __name__ == '__main__':
    main()
