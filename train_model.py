import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


np.random.seed(42)

# Generate synthetic student data
n = 300

attendance = np.random.randint(50, 101, n)
internal_marks = np.random.randint(30, 101, n)
assignment_marks = np.random.randint(30, 101, n)
previous_score = np.random.randint(30, 101, n)

# Create target variable
result = (
    (attendance >= 75)
    & (internal_marks >= 40)
    & (assignment_marks >= 40)
    & (previous_score >= 40)
).astype(int)

X = np.column_stack(
    (
        attendance,
        internal_marks,
        assignment_marks,
        previous_score,
    )
)

y = result

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Model trained successfully")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# Save model
joblib.dump(model, "student_result_model.pkl")

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist(),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model saved as student_result_model.pkl")
print("Metrics saved as metrics.json")
