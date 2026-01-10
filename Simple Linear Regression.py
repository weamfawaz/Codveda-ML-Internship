import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO",
    "B", "LSTAT", "PRICE"
]

df=pd.read_csv(r"C:\Users\Elhussien\.spyder-py3\house Prediction Data Set.csv",delim_whitespace=True,header=None,names=columns)
print(df.head())
print(df.info())
print(df.describe())
#filling missing value 
df.fillna(df.mean(),inplace=True)
#Feature&Target
x=df[['RM']]
y=df['PRICE']
#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#train model
model=LinearRegression()
model.fit(x_train,y_train)
#predict model
y_pred=model.predict(x_test)

print('MSE:', mean_squared_error(y_test,y_pred))
print('R2:',r2_score(y_test, y_pred))


plt.scatter(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, linewidth=2, label="Predicted")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

