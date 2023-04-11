import pandas as pd
from locust import HttpUser, task

test_data = pd.read_csv("/input/train.csv.zip", usecols=[3])

class PredictUser(HttpUser):
    host="http://api:5000"

    @task
    def predict(self):
        self.client.post("/predict", json={"text": test_data.sample(1).values[0, 0]})
