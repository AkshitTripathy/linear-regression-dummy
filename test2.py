import numpy as np #imports numpy and sets an alias as np
import pandas as pd #imports pandas and sets an alias as pd
from sklearn.linear_model import LinearRegression #imports Linear Regression Model from skikit learn
url = "data/test_data.csv" #Sets URL of the dataset from and to which is taken for setting and updating model
dataset = pd.read_csv(url, header=0) #uses pandas read_csv method to fetch dataset from url and sets header 1st line as names
X = dataset['X'].to_numpy() #gets X column from dataset, converts it to numpy array using to_numpy() and saves it to X
Y = dataset['Y'].to_numpy() #gets Y column from dataset, converts it to numpy array using to_numpy() and saves it to Y
model = LinearRegression().fit(X.reshape((-1, 1)),Y) #Fits X and Y to LinearRegression Model
A_input = float(input("Enter the value of x = ")) #takes input from user for value of x to predict value of y
A = np.array([A_input]).reshape((-1, 1)) #Creates array of value taken from the user
B = model.predict(A) #Predicts Value of y using model and value taken from user
#model.intercept_ + model.coef_ * A
print('Predicted Value is ',B) #prints predicted value
input_dataset = pd.DataFrame({'X': A_input, 'Y': B}) #Creates a pandas dataframe of x value from user input and y value from predicted value and stores it into input_dataset
input_dataset.to_csv(url, mode='a', header=False, index=False) #appends input_dataset to same dataset from url using to_csv function and removes header and index