This repository contains complete ecosystem for training and deploying the model for scoring the readability of the text

It consists of five services (in order of execution): 
    - train (where finetuning happens)
    - optimization (some optimizations applied and model is converted to onnx format)
    - inference (deployong the model to triton server)
    - api (exposing the endpoint for making post requests)
    - load_testing  (sending ~13k requests to measure the performance of the inference)

To run everything you just need to do `docker compose up`. All services will be started and killed in the correct order. At the very end only `api` service will remain running. If you don't need some services - just comment the corresponding lines in `docker-compose.yml`

Before you run - you'll need to place a `train.csv.zip` (from https://www.kaggle.com/c/commonlitreadabilityprize/overview) here - [data/input/](data/input/)

### Training ([train](services/train/))
`BertForSequenceClassification` was chosen for finetuning. After 3 epochs of fine-tuning it was able to achieve
Training script is [here](services/train/src/main.py)
The local notebook is [here](services/train/notebooks/train.ipynb), the interactive kaggle version is [here](https://www.kaggle.com/natenten/readability-scoring)
After three epochs validation loss looks like this:
```
+-------+---------------+-----------------+----------+
| Epoch | Training Loss | Validation Loss |   Rmse   |
+-------+---------------+-----------------+----------+
|     1 | No log        |        0.673124 | 0.678524 |
|     2 | No log        |        0.683692 | 0.690056 |
|     3 | No log        |        0.649549 | 0.655264 |
+-------+---------------+-----------------+----------+
```

### Optimization ([optimization](services/optimization/))
To optimize the model a bit and to deploy it to triton server we conver it to onnx format using huggingface `optimum` package. The corresponding [Dockerfile](services/optimization/Dockerfile)

### Inference ([inference](services/inference/))
To make a model serving a bit more reliable and to get the access to different metrics (Prometheus) we'll deploy our model to Triton Inference Server - [config](services/inference/model/config.pbtxt)

### API ([api](services/api/))
To expose our model to extternal world we'll create a FastAPI application. To get a readability score one needs to make a POST request to `/predict` endpoint with json containing `text` field.
Curl example
```bash
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{"text":"Some text to score"}'
```
Response example:
```
["Some text to score",-0.06885720044374466]
```

### Load testing ([load_testing](services/load_testing/))
This service will take a training dataset and wil send requests with texts from it for 2 minutes using `locust` library
Here are the results for performing the inference on RTX 3060. As we can see it is possible to achieve ~110 requests per second (~2 rps for i7 3770)
```
| Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
| --------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
| POST     /predict                                                                       13120     0(0.00%) |   1658      72    3685   1700 |  109.88        0.00
| --------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
|          Aggregated                                                                     13120     0(0.00%) |   1658      72    3685   1700 |  109.88        0.00
|
|
| Response time percentiles (approximated)
| Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
| --------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
| POST     /predict                                                                             1700   2000   2100   2100   2700   2800   2800   2800   3100   3700   3700  13120
| --------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
|          Aggregated                                                                           1700   2000   2100   2100   2700   2800   2800   2800   3100   3700   3700  13120
```
