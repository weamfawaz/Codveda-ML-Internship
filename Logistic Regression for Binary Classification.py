import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc

data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target # 0 = malignant, 1 = benign

print(x.head())
print(pd.Series(y).value_counts())
#scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
#split data to train & test
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
#train model
model=LogisticRegression()
model.fit(x_train,y_train)
#predict model
y_pred=model.predict(x_test)
#evaluate model
print('Accuracy_score: ', accuracy_score(y_test, y_pred))
print('Confusion_matrix: ', confusion_matrix(y_test, y_pred))
#roc_curve
y_prob = model.predict_proba(x_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
#coeffecient
coeff_df = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print(coeff_df.head())
