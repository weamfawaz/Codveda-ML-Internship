import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
#data preprocessing
df=pd.read_csv(r"C:\Users\Elhussien\.spyder-py3\1) iris.csv")
print(df.head())
print(df.info())
# split features & target
x=df.drop('species',axis=1)
y=df['species']
#encoding 
le=LabelEncoder()
y=le.fit_transform(y)
#scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
#split data
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
#train model
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_train,y_train)
#predict model
y_pred=KNN.predict(x_test)
#evaluate model
print('accuracy:',accuracy_score(y_test, y_pred,))
print('Classification Report:\n', classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#Use different values of K and compare the results.
k_values=range(1,11)
accuracies=[]
for k in k_values:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred_k=model.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))
plt.plot(k_values,accuracies,marker='o')    
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.show()
