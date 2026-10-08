import os
import unittest
import json
import joblib
import numpy as np


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run the training pipeline before testing
        os.system("python train_model.py")

        cls.model = joblib.load("student_result_model.pkl")

        with open("metrics.json", "r") as f:
            cls.metrics = json.load(f)

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_threshold(self):
        self.assertGreaterEqual(self.metrics["accuracy"], 0.70)

    def test_confusion_matrix_shape(self):
        cm = np.array(self.metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (2, 2))

    def test_prediction_pass(self):
        prediction = self.model.predict([[90, 80, 85, 88]])
        self.assertEqual(prediction[0], 1)

    def test_prediction_fail(self):
        prediction = self.model.predict([[50, 30, 35, 40]])
        self.assertEqual(prediction[0], 0)

    def test_model_has_predict_method(self):
        self.assertTrue(hasattr(self.model, "predict"))


if __name__ == "__main__":
    unittest.main()
