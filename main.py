import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Pycharm projects\ML\lrdata.csv')
df.head()

y = df['time']
x = df[['cases', 'increasingrate', 'decreasingrate']]


def CasesReg(x, y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x, y)
    return reg


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

reg = CasesReg(x_train, y_train)


y_pred = reg.predict(x_test)
y_pred


efficiency=reg.score(x_test, y_test)
print('Efficiency:',efficiency)

plt.plot(x, reg.predict(x), '*')
plt.legend(labels=['cases', 'increasingrate', 'decreasingrate'])
plt.xlabel("Cases Rate")
plt.ylabel("Time")
plt.show()
