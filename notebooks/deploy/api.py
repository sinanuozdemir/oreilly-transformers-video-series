from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
import os
from transformers import pipeline

app = FastAPI()

print("loading tokenizer + model")
CLF = pipeline(
    'text-classification', 'profoz/distilbert-toxic-classifier',
    # specify commit in HF. Usually you'd want to use an environment variable here
    # If you don't set this it will simply use the most recent version
    # revision='GIT COMMIT HASH',
    use_fast=True,  return_all_scores=True,
    use_auth_token=os.environ.get('HUGGING_API_KEY')  # if we need an API key / if our model is private on HF
)
print("loaded tokenizer + model")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    probabilities: Dict[str, float]
    label: str
    confidence: float

@app.post("/predict", response_model=Response)
def predict(request: Request):
    output = sorted(CLF(request.text)[0], key=lambda x: x['score'], reverse=True)  # use our pipeline and sort results
    return Response(
        label=output[0]['label'], confidence=output[0]['score'],
        probabilities={item['label']: item['score'] for item in output}
    )
