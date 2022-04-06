from hashlib import md5
from sklearn import tree
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_squared_error


# N = 1048575
N = 10
M = 9

def MSE(y_true, y_pred):
    """
    Returns the Mean Squared Error
    """
    return ((y_true - y_pred) ** 2).mean()

def get_data():
    """
    Returns the data from the xlsx file
    """
    file_name = 'yeast.csv' 
    df = pd.read_csv(file_name, index_col=0, header=None)
    # print(df.head()) # print the first 5 rows
    return df
md = get_data()
md.dropna(inplace = True)
print(md.shape)
md.replace('?', 0, inplace = True)

# print(md)

trainNo = 1000
X = md.iloc[:, 0:8].values.reshape(-1, 8)  # values converts it into a numpy array
Y = md.iloc[:, 8:9].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
# print(X)

# print(type(Y[0]))

# print(Y)


# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression

# Y_pred = linear_regressor.predict(X[trainNo:])  # make predictions
# print(linear_regressor.coef_, "\n")  # prints the coefficients


# print(linear_regressor.get_params())

# print(mean_squared_error(Y_pred, Y[trainNo:]) / ((777384 +  776950 + 788567 + 678984) / 4))

clf = LogisticRegression(random_state=0).fit(X[:trainNo], Y[:trainNo])
predict = clf.predict(X[trainNo:])
print('proba: ', clf.predict_proba(X[trainNo:]))
print('mean accuracy: ', clf.score(X[trainNo:], Y[trainNo:]))
print('estimates', predict)
# print('params', clf.get_params())
# plt.scatter(predict, Y[trainNo:])
# plt.show()


# xlsx_file = Path('upd.x')
# wb_obj = openpyxl.load_workbook(xlsx_file)
# sheet = wb_obj.active

# # x = np.empty([N, M])
# x = np.empty([N, M])
# y = np.empty([N, 3])

# # print(x)

# # print(type(sheet))

# # print(sheet['A1'].value)

# i = 0

# for row in sheet.iter_rows():
#     if (type(row[0].value) != datetime):
#         continue
#     # print(i)
#     print(row[2].value)
#     if row[2].value == '?' or row[3].value == '?' or row[4].value == '?' or row[5].value == '?' or row[6].value == '?' or row[7].value == '?':
#             print("cont")
#             continue
#     k = 0
#     l = 0
#     for cell in row:
#         if cell.column == 1:
#              #print(cell.value, cell.value.__class__)
            
#             x[i][k] = int(cell.value.year)
#             x[i][k+1] = int(cell.value.month)
#             x[i][k+2] = int(cell.value.day)
#             k = k + 3
#         elif cell.column == 2:
#             # print(cell.value, cell.value.__class__)
#             x[i][k] = int(cell.value.hour)
#             x[i][k + 1] = int(cell.value.minute)
#             k = k + 2
#         elif cell.column > 6:
#             y[i][l] = float(cell.value)
#             l = l + 1
#         else:
#             x[i][k] = float(cell.value)
#             k = k + 1
#     i = i + 1
#     # print()
# y1, y2, y3 = np.empty([i]), np.empty([i]), np.empty([i])

# # print(i)
# print('length', len(x))

# for k in range(i):
#     y1[k] = y[k][0]
#     y2[k] = y[k][1]
#     y3[k] = y[k][2]
# # np.reshape(x, [-1, i])

# # print()
# # print(type(x[0]))
# # print(type(x))
# x = x[0: i]
# y = y[0: i]
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(x, y3)  # perform linear regression
# Y_pred = linear_regressor.predict(x)  # make predictions
# print('error', MSE(y3, Y_pred) / i)  # prints the mean squared error

# print(linear_regressor.coef_, "\n")  # prints the coefficients
# print(linear_regressor.get_params())


# print('\n', x)

# print(y)
# print(y1)
# print(y2)
# print(y3)


# clf = tree.DecisionTreeRegressor()

