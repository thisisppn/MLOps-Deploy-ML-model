# Put the code for your API here.
import pickle
from typing import Optional

import numpy as np
import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()


@app.get("/")
def read_root():
    return "Welcome to the API"


class Item(BaseModel):
    age: int
    fnlgt: int
    education_num: int = Field(alias='education-num')
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')

    workclass: str
    education: str
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias='native-country')


@app.post("/infer/")
async def infer(item: Item):

    # separate the categorical and normal values
    item_dict = item.dict(by_alias=True)

    X = pd.DataFrame.from_records([item_dict])

    encoder = pickle.load(open("model/encoder.pkl", 'rb'))
    model = pickle.load(open("model/naive_model.pkl", 'rb'))

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        X, categorical_features=cat_features, training=False, encoder=encoder,
    )

    pred = inference(model, X_test)

    labels = ["<=50K", ">50L"]

    return labels[pred[0]]

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)