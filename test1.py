import numpy as np #imports numpy
from sklearn.linear_model import LinearRegression #imports Linear Regression Model from skikit learn
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) #Creates an array of 6 elements, reshapes it to y axis and saves it in X
y = np.array([5, 20, 14, 32, 22, 38]) #Creates an array of 6 elements using numpy
model = LinearRegression().fit(x,y) #Fits LinearRegression Model to X and Y
r_sq = model.score(x, y) #Calculates R2(square) of Linear Regression model
A = float(input("Enter the value of x")) #takes input from user for value of x to predict value of y
A = np.array([A]).reshape((-1, 1)) #Creates array of value taken from the user
B = model.predict(A) #Predicts Value of y using model and value taken from user
#model.intercept_ + model.coef_ * A 
print('Predicted Value as per defined Linear Regression Model is',B)
