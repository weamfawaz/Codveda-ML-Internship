import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,f1_score

#load dataset
iris=load_iris()
#split data feature & target
x=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target

print(x.head())
print(pd.Series(y).value_counts())

#split train test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)

#predit model
y_pred=dt_model.predict(x_test)

#evaluate model
print('accuracy_score: ',accuracy_score(y_test, y_pred))
print('f1_score: ',f1_score(y_test, y_pred,average='weighted'))

#Visualize the Decision Tree
plt.figure(figsize=(16,8))
plot_tree(dt_model, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names,
          filled=True)
plt.title("Decision Tree Classifier (Before Pruning)")
plt.show()

#Pruning the Tree (Prevent Overfitting)
dt_purned=DecisionTreeClassifier(max_depth=3,random_state=42)
dt_purned.fit(x_train, y_train)

y_pred_pruned = dt_purned.predict(x_test)


print('accuracy_score: ',accuracy_score(y_test, y_pred_pruned))
print('f1_score: ',f1_score(y_test, y_pred_pruned,average='weighted'))

plt.figure(figsize=(16,4))
plot_tree(dt_purned,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True
)
plt.title("Decision Tree Classifier (After Pruning)")
plt.show()
