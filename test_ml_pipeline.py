import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Run the training pipeline
        os.system("python train_model.py")

        cls.model = joblib.load(
            "student_result_model.pkl"
        )

        with open("metrics.json", "r") as file:
            cls.metrics = json.load(file)

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists(
                "student_dataset_10000_rows.csv"
            )
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists(
                "student_result_model.pkl"
            )
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists(
                "metrics.json"
            )
        )

    def test_accuracy_is_valid(self):
        accuracy = self.metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    def test_confusion_matrix_shape(self):
        matrix = self.metrics["confusion_matrix"]

        self.assertEqual(
            len(matrix),
            2
        )

        self.assertEqual(
            len(matrix[0]),
            2
        )

        self.assertEqual(
            len(matrix[1]),
            2
        )

    def test_model_prediction(self):

        sample = pd.DataFrame([{
            "study_hours": 10,
            "attendance": 95,
            "assignments_completed": 20,
            "previous_score": 90
        }])

        prediction = self.model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )

    def test_high_performance_prediction(self):

        sample = pd.DataFrame([{
            "study_hours": 10,
            "attendance": 95,
            "assignments_completed": 20,
            "previous_score": 90
        }])

        prediction = self.model.predict(sample)[0]

        self.assertEqual(
            int(prediction),
            1
        )


if __name__ == "__main__":
    unittest.main()
