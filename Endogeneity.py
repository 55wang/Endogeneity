import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor


sample_num = 10000
z1 = np.random.normal(0,1,sample_num)
z2 = np.random.normal(0,1,sample_num)
x = np.random.normal(0,1,sample_num) - 20*z1
y = x + 100*z1 - z2 + np.random.normal(0,1,sample_num)
z3 = x - 2*y + np.random.normal(0,1,sample_num)

print('Correlation: ')
print(np.corrcoef(x, z1))
print()

X = pd.DataFrame({'const':np.array([1]*len(y)),'x':x,'z1':z1})
Y = np.array([y]).T
model = OLS(Y,X)
result = model.fit()
print('independent variable: x + z1')
print(result.summary())
print()

X = pd.DataFrame({'const':np.array([1]*len(y)),'x':x,'z1':z1,'z2':z2})

Y = np.array([y]).T
model = OLS(Y,X)
result = model.fit()
print('independent variable: x + z1 + z2')
print(result.summary())
print()

X = pd.DataFrame({'const':np.array([1]*len(y)),'x':x})
Y = np.array([y]).T
model = OLS(Y,X)
result = model.fit()
print('independent variable: x')
print(result.summary())
print()

X = pd.DataFrame({'const':np.array([1]*len(y)),'x':x,'z1':z1,'z3':z3})
Y = np.array([y]).T
model = OLS(Y,X)
result = model.fit()
print('independent variable: x + z1 + z3')
print(result.summary())
print()

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('VIF:')
print(X.columns)
print(vif)
