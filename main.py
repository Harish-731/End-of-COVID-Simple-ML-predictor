import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Importing and reading file using pandas
df = pd.read_csv(r'D:\Pycharm projects\ML\lrdata.csv')
df.head()

# Using multi linear regression as the values are continuous where x are independent
# values and y : dependent values
y = df['time']
x = df[['cases', 'increasingrate', 'decreasingrate']]


# Function named CasesReg for using linear regression model and training the model
# with given values
def CasesReg(x, y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x, y)
    return reg


# train_test_split is a function in Sklearn model selection for splitting data
# arrays into two subsets: for training data and for testing data.
# With this function, you don't need to divide the dataset manually.
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

# Passing training data to my Linear regression model
reg = CasesReg(x_train, y_train)

# Using predict to get predicted values of y
y_pred = reg.predict(x_test)
y_pred

# Calculating efficiency. It internally calculates y_pred again and gives the
# efficiency
efficiency=reg.score(x_test, y_test)
print('Efficiency:',efficiency)
# Plotting graph using matplotlib.
plt.plot(x, reg.predict(x), '*')
plt.legend(labels=['cases', 'increasingrate', 'decreasingrate'])
plt.xlabel("Cases Rate")
plt.ylabel("Time")
plt.show()
