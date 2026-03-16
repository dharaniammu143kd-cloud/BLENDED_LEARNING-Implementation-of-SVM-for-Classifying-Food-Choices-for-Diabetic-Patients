# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Import the required libraries and load the food items dataset.
2. Select features and target variable, then split the dataset into training and testing data.
3. Scale the data and train the Support Vector Machine (SVM) model using GridSearchCV for hyperparameter tuning.
4. Predict the results, evaluate accuracy, and display the confusion matrix.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary (1).csv')
print(data.head())
print(data.columns)

features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()
param_grid = {
'C': [0.1, 1, 10, 100],
'kernel': ['linear', 'rbf'],
'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Name: DHARANI B")
print("Register Number: 212225230053")
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: DHARANI B")
print("Register Number: 212225230053")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
<img width="579" height="600" alt="image" src="https://github.com/user-attachments/assets/6332fe73-6bd0-47c0-986b-ab5feb2a943d" />
<img width="634" height="585" alt="image" src="https://github.com/user-attachments/assets/552b30ca-6733-4290-a39b-cb22aa099368" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
