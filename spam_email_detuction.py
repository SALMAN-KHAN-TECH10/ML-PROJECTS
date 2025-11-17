import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"d:\JUPYTER\PYTHON-PROJECTS\spambase_csv.csv")
# Clean the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Optional: Label mapping
data['class'] = data['class'].replace([1, 0], ['Not Spam', 'Spam'])
# Split into features and labels
X = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]   # labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the SVM model
print('your model training prcess is starts')
print('Training.........')
model = SVC()
model.fit(X_train, y_train)
print('now your model is trained ')
# Predict and evaluate
print('you model is preducting data............')
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Model Accuracy:", accuracy)
# Plot a subset for better visualization
num_points = 50 
plt.figure(figsize=(16, 3))

plt.plot(range(num_points), y_test[:num_points].values, label='Actual', marker='o')
plt.plot(range(num_points), y_pred[:num_points], label='Predicted', marker='x')
plt.title("Actual vs Predicted Labels (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 3))
plt.scatter(range(50), y_test[:50], label='Actual', color='green', marker='o')
plt.scatter(range(50), y_pred[:50], label='Predicted', color='red', marker='x')
plt.title("Dot Graph - Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.show()
