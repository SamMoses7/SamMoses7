import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data creation (replace this with real data)
data = {
    'age': [25, 45, 35, 50, 23, 40, 35, 55, 40, 50],
    'gender': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'blood_pressure': [120, 130, 140, 150, 110, 135, 145, 160, 155, 140],
    'cholesterol': [180, 195, 200, 230, 190, 200, 210, 220, 200, 205],
    'diabetes': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'target': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop(columns='target')
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data creation (replace this with real data)
data = {
    'age': [25, 45, 35, 50, 23, 40, 35, 55, 40, 50],
    'gender': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'blood_pressure': [120, 130, 140, 150, 110, 135, 145, 160, 155, 140],
    'cholesterol': [180, 195, 200, 230, 190, 200, 210, 220, 200, 205],
    'diabetes': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'target': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop(columns='target')
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
