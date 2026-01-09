import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
df=pd.read_csv(r"C:\Users\Elhussien\.spyder-py3\1) iris.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
# split features & target
x = df.drop("species", axis=1)
y = df["species"]
#encoding
le=LabelEncoder()
y=le.fit_transform(y)
# scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
# train test split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)