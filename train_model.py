```python
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = "student_dataset_10000_rows.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "previous_score"
]

TARGET = "placement_status"


def train_model():
    print("Loading dataset...")

    data = pd.read_csv(DATASET_FILE)

    print("Dataset loaded successfully.")

    # Save demonstration dataset for GitHub Actions artifacts
    data.to_csv("student_results.csv", index=False)
    print("Dataset saved as student_results.csv")

    print("Number of records:", len(data))
    print("Columns:", list(data.columns))

    # Check for missing values
    if data[FEATURES + [TARGET]].isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")

    # Convert target to binary values
    # Not Placed = 0, Placed = 1
    data[TARGET] = data[TARGET].map({
        "Not Placed": 0,
        "Placed": 1
    })

    if data[TARGET].isnull().any():
        raise ValueError("Unexpected target value found.")

    X = data[FEATURES]
    y = data[TARGET]

    print("\nFeatures used:")
    print(FEATURES)

    print("\nTarget distribution:")
    print(y.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Machine Learning pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("\nTraining Logistic Regression model...")
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    # Save trained model
    joblib.dump(model, "student_result_model.pkl")
    print("\nModel saved as student_result_model.pkl")

    # Save evaluation metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "confusion_matrix": matrix.tolist()
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    print("\nAll ML artifacts generated successfully.")


if __name__ == "__main__":
    train_model()
```
