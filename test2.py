import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
url = "data/test_data.csv"
dataset = pd.read_csv(url, header=0)
X = dataset['X'].to_numpy()
Y = dataset['Y'].to_numpy()
model = LinearRegression().fit(X.reshape((-1, 1)),Y)
A_input = int(input("Enter the value of x = "))
A = np.array([A_input]).reshape((-1, 1))
B = model.predict(A) #model.intercept_ + model.coef_ * A
print('Predicted Value is ',B)
input_dataset = pd.DataFrame({'X': A_input, 'Y': B})
input_dataset.to_csv(url, mode='a', header=False, index=False)
