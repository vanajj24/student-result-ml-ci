import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "study_hours": 8,
                "attendance": 95,
                "assignments_completed": 10,
                "previous_score": 90
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"], "PLACED"
        )

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "study_hours": 1,
                "attendance": 40,
                "assignments_completed": 1,
                "previous_score": 20
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"], "NOT PLACED"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"study_hours": 8, "attendance": 95}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
