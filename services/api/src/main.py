import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from transformers import BertTokenizerFast
import tritonclient.http as httpclient

from typing import Tuple

app = FastAPI()

tokenizer = BertTokenizerFast.from_pretrained("/tokenizer", return_tensors='pt')
triton_client = httpclient.InferenceServerClient(url='inference:8000')

def preprocess(text:str, tokenizer:BertTokenizerFast) -> dict:
    """Clean and tokenize the text
    Parameters
    ----------
    text : str
        Text to tokenize
    tokenizer : AutoTokenizer
        Approptiate tokenizer from the huggingface
    Returns
    -------
    dict
        Dictionary with "input_ids", "attention_mask" and "token_type_ids" keys
    """
    return tokenizer(text, padding='max_length', truncation=True, max_length=300, return_tensors="np")

def get_prediction(text:str) -> float:
    """Send the text to a triton inference server and get the prediction
    Parameters
    ----------
    text : str
        Text to evaluate
    Returns
    -------
    float
        Readability score
    """
    tokenized_data = preprocess(text, tokenizer)

    inputs = [0, 0, 0]
    inputs[0] = httpclient.InferInput('input_ids', [1, 300], 'INT64')
    inputs[1] = httpclient.InferInput('attention_mask', [1, 300], 'INT64')
    inputs[2] = httpclient.InferInput('token_type_ids', [1, 300], 'INT64')

    # Initialize the data
    inputs[0].set_data_from_numpy(tokenized_data['input_ids'], binary_data=False)
    inputs[1].set_data_from_numpy(tokenized_data['attention_mask'], binary_data=False)
    inputs[2].set_data_from_numpy(tokenized_data['token_type_ids'], binary_data=False)

    output = httpclient.InferRequestedOutput('logits',  binary_data=False)
    logits = triton_client.infer('readability', model_version='1', inputs=inputs, outputs=[output])
    scores = logits.as_numpy('logits')

    return scores[0]

@app.get("/alive")
def alive() -> bool:
    return True

@app.post("/predict")
async def predict(request: Request) -> Tuple[str, float]:
    """Function that handles incoming post request for the "/predict" endpoint
    Parameters
    ----------
    request : Request
        Request object containing request parameters as json
    Returns
    -------
    tuple[str, float]
        str - text
        float - readability score
    """
    json = await request.json()

    text = json['text']

    return text, get_prediction(text)

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=5000, workers=10)